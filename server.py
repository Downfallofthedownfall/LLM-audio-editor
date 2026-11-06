"""
音频自动去重/去空 Web 界面（Flask + SSE 实时进度）
- 定位说话内容并拼接：只保留说话片段，切掉空隙，保持原始采样率/声道
- 可选 LLM(OpenAI 兼容, 任意 chat.completions / responses 端点) 智能判重
- 输出可选 wav（无损）/ mp3（高码率）
"""
import os, uuid, json, time, re, threading, traceback, shutil, tempfile, glob
import sys
from urllib.parse import urlsplit
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from flask import Flask, request, send_from_directory, Response

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, ".hf-cache"))

from dedup import dedup_audio, new_job_dir, output_extension, WORK_DIR

app = Flask(__name__)
OUT_DIR = os.path.join(_HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 上传缓存等临时文件放在项目所在的磁盘上，不占系统盘；子进程也继承这个设置
TMP_DIR = os.path.join(WORK_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = TMP_DIR
tempfile.tempdir = TMP_DIR

_jobs: dict = {}
_jobs_lock = threading.Lock()

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".mp4", ".aac")


def _clean_stale_work() -> None:
    """启动时清掉上次被强行关闭留下的中间文件。只动 .work，不碰 output 和用户文件。"""
    for path in glob.glob(os.path.join(WORK_DIR, "job_*")):
        shutil.rmtree(path, ignore_errors=True)
    for name in os.listdir(TMP_DIR):
        path = os.path.join(TMP_DIR, name)
        try:
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        except OSError:
            pass


def _safe_name(name: str) -> str:
    name = os.path.basename(name or "").strip()
    return re.sub(r"[^\w.\-]", "_", name)


def _out_path(input_filename: str, user_outname: str, ext: str) -> str:
    """output 里的目标路径：扩展名用真实格式，音频和同名 AAF 都不覆盖已有文件。"""
    stem = _safe_name(user_outname) if user_outname and user_outname.strip() else ""
    if stem.lower().endswith(_AUDIO_EXTS):
        stem = os.path.splitext(stem)[0]
    if not stem:
        stem = _safe_name(os.path.splitext(os.path.basename(input_filename))[0]) + "_deduped"
    candidate, i = stem, 0
    while (os.path.exists(os.path.join(OUT_DIR, candidate + ext))
           or os.path.exists(os.path.join(OUT_DIR, candidate + ".aaf"))):
        i += 1
        candidate = f"{stem}_{i}"
    return os.path.join(OUT_DIR, candidate + ext)


def _is_local_request() -> bool:
    """只接受本机页面发起的请求：本地路径模式能读取任意本机文件，不能让其他网页触发。"""
    if (request.host.rsplit(":", 1)[0]) not in ("127.0.0.1", "localhost"):
        return False
    origin = request.headers.get("Origin")
    return not origin or urlsplit(origin).netloc == request.host


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Audio Dedup</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:40px auto;padding:0 20px;color:#222}
 h1{font-size:24px} label{display:block;margin:14px 0 6px;font-weight:600;font-size:14px}
 input,select{display:block;width:100%;padding:8px 10px;box-sizing:border-box;border:1px solid #ccc;border-radius:8px;font-size:14px}
 input[type=checkbox]{width:auto;display:inline-block;margin-right:6px}
 button{margin-top:18px;width:100%;padding:12px;background:#2563eb;border:none;border-radius:8px;color:#fff;font-size:15px;cursor:pointer}
 button:hover{background:#1d4ed8} button:disabled{background:#9ca3af;cursor:not-allowed}
 .muted{color:#666;font-size:12px;margin-top:6px}
 .bar-wrap{height:12px;background:#e5e7eb;border-radius:6px;margin-top:14px;overflow:hidden}
 .bar{height:100%;width:0;background:#2563eb;transition:width .3s}
 #status{margin-top:10px;font-size:14px;white-space:pre-wrap}
 .done{color:#059669;font-weight:600} .err{color:#dc2626}
 a{color:#2563eb} audio{width:100%;margin-top:12px;border-radius:8px}
 .row{display:flex;gap:12px} .row>div{flex:1}
</style></head><body>
<h1>🎙️ Speech Extraction + Dedup</h1>
<p class="muted">Keeps only the speech segments and splices them together (automatic silence/pause removal), preserving the original sample rate and channels. Your input file is never modified.</p>

<div id="formwrap">
<form id="f">
 <label>Local file path (recommended for large files: read in place, no upload or copy)</label><input name="local_path" placeholder="F:\\videos\\talk.mp4">
 <label>Or upload a file</label><input type="file" name="audio" accept="audio/*,video/*">
 <label>OpenAI-compatible API URL</label><input name="api_url" placeholder="https://api.openai.com/v1">
 <label>Model name</label><input name="model" placeholder="gpt-4o-mini">
 <label>API key</label><input name="api_key" type="password">
 <label>API type</label><select name="api_mode">
   <option value="auto" selected>Auto (try Chat Completions, then Responses)</option>
   <option value="chat">Chat Completions</option>
   <option value="responses">Responses</option>
 </select>
 <label>Audio language</label><select name="language">
   <option value="en" selected>English</option>
   <option value="zh">中文 (Chinese)</option>
 </select>
 <div class="row">
   <div><label>Whisper model</label>
     <select name="whisper"><option value="large" selected>large (default, most accurate)</option><option value="small">small (faster)</option></select>
   </div>
   <div><label>Silence gap (seconds, larger = keep more pauses, cut less)</label><input name="keep_gap" type="number" value="0.8" step="0.1" min="0.1" max="3"></div>
    <div><label>Pause split sensitivity (seconds, smaller = more granular)</label><input name="pause_dur" type="number" value="0.25" step="0.05" min="0.1" max="1.5"></div>
    
 </div>
 <div class="row">
   <div><label>Output format</label><select name="out_format"><option value="auto" selected>Match source</option><option value="wav">WAV (lossless)</option><option value="flac">FLAC (lossless)</option><option value="mp3">MP3</option></select></div>
   <div><label>MP3 bitrate</label><select name="out_bitrate"><option value="192k">192k</option><option value="320k" selected>320k</option><option value="128k">128k</option></select></div>
 </div>
 <label><input type="checkbox" name="use_llm" value="1"> Use LLM intelligent dedup (remove duplicate / redundant lines)</label>
 <label>Output file name (optional, default &lt;name&gt;_deduped.wav)</label><input name="outname" placeholder="e.g. my_clean.wav">
 <button id="btn" type="submit">Start Processing</button>
</form>
</div>

<div id="proc" style="display:none">
 <div class="bar-wrap"><div class="bar" id="bar"></div></div>
 <div id="status"></div>
 <div id="res"></div>
</div>

<script>
document.getElementById('f').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const btn = document.getElementById('btn'); btn.disabled = true;
  document.getElementById('formwrap').style.display='none';
  const proc = document.getElementById('proc'); proc.style.display='block';
  const fd = new FormData();
  for (const [k,v] of new FormData(document.getElementById('f'))) fd.append(k,v);
  const usePath = fd.get('local_path').trim() !== '';
  if (usePath) fd.delete('audio');
  document.getElementById('status').innerText = usePath ? 'Starting…' : 'Uploading…';
  try {
    const r = await fetch('/dedup', {method:'POST', body: fd});
    const d = await r.json();
    if (d.error){ showErr(d.error); return; }
    const es = new EventSource('/progress/' + d.job_id);
    es.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      document.getElementById('bar').style.width = (m.progress*100)+'%';
      document.getElementById('status').innerText = m.msg;
      if (m.done){
        es.close();
        if (m.error){ showErr(m.error); }
        else {
          document.getElementById('status').innerHTML = '<span class="done">✅ Done: kept '+m.kept_sec+'s of '+m.total_sec+'s, removed about '+m.removed_sec+'s.</span>';
          let resHtml = '<p><a href="/output/'+encodeURIComponent(m.outname)+'" download>⬇️ Download deduped audio</a></p><audio controls src="/output/'+encodeURIComponent(m.outname)+'"></audio>';
          if (m.out_aaf) { resHtml += '<p><a href="/output/'+encodeURIComponent(m.out_aaf)+'" download>🎬 Download Premiere sequence (AAF)</a> <span class="muted">Import the AAF → link source media (one source) → correct manually</span></p>'; }
          document.getElementById('res').innerHTML = resHtml;
          btn.disabled=false; document.getElementById('formwrap').style.display='block';
        }
      }
    };
  } catch(e){ showErr('Request failed: '+e); }
  function showErr(t){ document.getElementById('status').innerHTML='<span class="err">'+t+'</span>'; btn.disabled=false; document.getElementById('formwrap').style.display='block'; }
});
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/dedup", methods=["POST"])
def dedup():
    if not _is_local_request():
        return {"error": "Only requests from this machine's own page are accepted."}, 403
    # “复制文件地址”得到的路径带引号，这里去掉
    local_path = request.form.get("local_path", "").strip().strip('"\'').strip()
    f = request.files.get("audio")
    if local_path:
        if not os.path.isabs(local_path):
            return {"error": "Please enter the full path, e.g. F:\\videos\\talk.mp4"}, 400
        if not os.path.isfile(local_path):
            return {"error": "File not found. Please check the path."}, 400
    elif not (f and f.filename):
        return {"error": "Please enter a local file path or choose a file to upload."}, 400
    api_url = request.form.get("api_url", "").strip()
    model = request.form.get("model", "").strip()
    api_key = request.form.get("api_key", "").strip()
    api_mode = request.form.get("api_mode", "auto").strip() or "auto"
    whisper = request.form.get("whisper", "large")
    keep_gap = float(request.form.get("keep_gap", "0.8") or 0.8)
    pause_dur = float(request.form.get("pause_dur", "0.25") or 0.25)
    seg_gap = float(request.form.get("seg_gap", "0.3") or 0.3)
    out_format = request.form.get("out_format", "auto")
    out_bitrate = request.form.get("out_bitrate", "320k")
    use_llm = request.form.get("use_llm") == "1"
    language = request.form.get("language", "en").strip() or "en"
    if language not in ("en", "zh"):
        language = "en"
    outname = request.form.get("outname", "")

    if api_mode not in ("auto", "chat", "responses"):
        api_mode = "auto"
    if use_llm and not (api_url and model and api_key):
        return {"error": "Please fill in the API URL, model name, and API key."}, 400

    # 所有中间文件都在 job_dir，结束时整个删除；本地路径模式的原文件不在其中，不会被动到
    job_dir = new_job_dir()
    try:
        if local_path:
            audio_path = local_path
        else:
            audio_path = os.path.join(job_dir, _safe_name(f.filename).lstrip(".") or "upload.bin")
            f.save(audio_path)
        out_path = _out_path(audio_path, outname, output_extension(audio_path, out_format))
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    out_url = os.path.basename(out_path)

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"progress": 0.0, "msg": "Queued…", "done": False, "error": None,
                         "kept_sec": 0.0, "removed_sec": 0.0, "total_sec": 0.0, "outname": out_url, "out_aaf": None}

    def _progress(frac, msg):
        with _jobs_lock:
            _jobs[job_id]["progress"] = frac
            _jobs[job_id]["msg"] = msg

    def worker():
        try:
            info = dedup_audio(audio_path, out_path, api_url=api_url, model=model, api_key=api_key,
                               whisper_model=whisper, language=language, keep_gap=keep_gap,
                               use_llm=use_llm, api_mode=api_mode, out_format=out_format, out_bitrate=out_bitrate,
                               seg_gap=seg_gap, pause_dur=pause_dur, progress=_progress, work_dir=job_dir)
            with _jobs_lock:
                j = _jobs[job_id]
                j["done"] = True
                j["kept_sec"] = info["kept_sec"]
                j["removed_sec"] = info["removed_sec"]
                j["total_sec"] = info["total_sec"]
                j["msg"] = "Done"
                j["progress"] = 1.0
                j["out_aaf"] = os.path.basename(info["out_aaf"]) if info.get("out_aaf") else None
        except Exception as e:
            traceback.print_exc()
            with _jobs_lock:
                _jobs[job_id]["done"] = True
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["msg"] = "Error"
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.route("/progress/<job_id>")
def progress(job_id):
    def gen():
        last = None
        while True:
            with _jobs_lock:
                j = _jobs.get(job_id)
            if j is None:
                yield "data: {}\n\n"
                break
            data = json.dumps({"progress": j["progress"], "msg": j["msg"], "done": j["done"],
                               "error": j["error"], "kept_sec": j["kept_sec"],
                               "removed_sec": j["removed_sec"], "total_sec": j["total_sec"],
                               "outname": j["outname"], "out_aaf": j["out_aaf"]})
            if data != last:
                last = data
                yield "data: " + data + "\n\n"
            if j["done"]:
                break
            time.sleep(0.3)
    return Response(gen(), mimetype="text/event-stream")


@app.route("/output/<path:name>")
def out_file(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    _clean_stale_work()
    app.run(host="127.0.0.1", port=7861, debug=False, threaded=True)
