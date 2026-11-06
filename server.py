"""
音频自动去重/去空 Web 界面（Flask + SSE 实时进度）
- 定位说话内容并拼接：只保留说话片段，切掉空隙，保持原始采样率/声道
- LLM(OpenAI 兼容, 任意 chat.completions / responses 端点) 智能判重
- 输出音频（wav / flac / mp3）和视频、音频都剪好的 Premiere 工程
"""
import os, uuid, json, time, re, threading, traceback, shutil, tempfile, glob, socket
import sys
from typing import Optional
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
HOST, PORT = "127.0.0.1", 7861
OUT_DIR = os.path.join(_HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 上传缓存等临时文件放在项目所在的磁盘上，不占系统盘；子进程也继承这个设置
TMP_DIR = os.path.join(WORK_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = TMP_DIR
tempfile.tempdir = TMP_DIR

_jobs: dict = {}
_jobs_lock = threading.Lock()
# 进行中任务预留的输出文件名（不含扩展名、统一小写），由 _jobs_lock 保护
_reserved_stems: set = set()
# 一次只跑一个任务：两个任务同时把模型加载到 GPU 容易显存不足，其余任务排队
_run_lock = threading.Lock()

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


def _claim_port() -> Optional[socket.socket]:
    """占住服务端口；已被占用（服务很可能已在运行）时返回 None。"""
    s = socket.socket()
    try:
        s.bind((HOST, PORT))
    except OSError:
        s.close()
        return None
    return s


def _safe_name(name: str) -> str:
    name = os.path.basename(name or "").strip()
    return re.sub(r"[^\w.\-]", "_", name)


def _out_path(input_filename: str, user_outname: str, ext: str) -> str:
    """output 里的目标路径：扩展名用真实格式，音频和同名 Premiere 工程都不覆盖已有文件。

    同时预留文件名，避免排队中或进行中的任务写到同一个文件；任务结束时用 _release_out_path 释放。
    """
    stem = _safe_name(user_outname) if user_outname and user_outname.strip() else ""
    if stem.lower().endswith(_AUDIO_EXTS):
        stem = os.path.splitext(stem)[0]
    if not stem:
        stem = _safe_name(os.path.splitext(os.path.basename(input_filename))[0]) + "_deduped"
    candidate, i = stem, 0
    with _jobs_lock:
        while (candidate.lower() in _reserved_stems
               or os.path.exists(os.path.join(OUT_DIR, candidate + ext))
               or os.path.exists(os.path.join(OUT_DIR, candidate + ".prproj"))
               or os.path.exists(os.path.join(OUT_DIR, candidate + ".xml"))):
            i += 1
            candidate = f"{stem}_{i}"
        _reserved_stems.add(candidate.lower())
    return os.path.join(OUT_DIR, candidate + ext)


def _release_out_path(path: str) -> None:
    with _jobs_lock:
        _reserved_stems.discard(os.path.splitext(os.path.basename(path))[0].lower())


def _is_local_request() -> bool:
    """只接受本机页面发起的请求：本地路径模式能读取任意本机文件，不能让其他网页触发。"""
    if (request.host.rsplit(":", 1)[0]) not in ("127.0.0.1", "localhost"):
        return False
    origin = request.headers.get("Origin")
    return not origin or urlsplit(origin).netloc == request.host


@app.before_request
def _only_this_machine():
    # 所有路由都检查：Host 校验挡住 DNS rebinding，否则别的网页能读到 output 里的结果
    if not _is_local_request():
        return {"error": "Only requests from this machine's own page are accepted."}, 403


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
 .done{color:#059669;font-weight:600} .err{color:#dc2626} .warn{color:#b45309;font-size:14px}
 a{color:#2563eb} audio{width:100%;margin-top:12px;border-radius:8px}
 .row{display:flex;gap:12px} .row>div{flex:1}
</style></head><body>
<h1>🎙️ Speech Extraction + Dedup</h1>
<p class="muted">Removes silences, mis-speaks and restarts, then gives you clean audio plus a Premiere project with the video and audio cut. Premiere opens briefly at the end of a run, so close it first. Your input file is never modified.</p>

<div id="formwrap">
<form id="f">
 <label>Local file path (recommended for large files: read in place, no upload or copy)</label><input name="local_path" placeholder="F:\\videos\\talk.mp4">
 <label>Or upload a file</label><input type="file" name="audio" accept="audio/*,video/*">
 <label>OpenAI-compatible API URL</label><input name="api_url" placeholder="https://api.deepseek.com" required>
 <label>Model name</label><input name="model" placeholder="deepseek-flash" required>
 <label>API key</label><input name="api_key" type="password" required>
 <label>API type</label><select name="api_mode">
   <option value="auto" selected>Auto (try Responses, then Chat Completions)</option>
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
 </div>
 <div class="row">
   <div><label>Output format</label><select name="out_format"><option value="flac" selected>FLAC (lossless)</option><option value="wav">WAV (lossless)</option><option value="auto">Match source (re-encodes if the source is lossy)</option><option value="mp3">MP3</option></select></div>
   <div><label>MP3 bitrate</label><select name="out_bitrate"><option value="192k">192k</option><option value="320k" selected>320k</option><option value="128k">128k</option></select></div>
 </div>
 <label>Output file name (optional, default &lt;name&gt;_deduped; the extension follows the output format)</label><input name="outname" placeholder="e.g. my_clean">
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
  const el = (id) => document.getElementById(id);
  const btn = el('btn'); btn.disabled = true;
  el('formwrap').style.display='none';
  el('proc').style.display='block';
  el('res').replaceChildren();
  el('bar').style.width = '0';
  const resetForm = () => { btn.disabled=false; el('formwrap').style.display='block'; };
  // 错误信息里可能有文件路径和 ffmpeg 日志，只当纯文本显示
  const showLine = (cls, text) => { const s = document.createElement('span'); s.className = cls; s.textContent = text; el('status').replaceChildren(s); };
  const showErr = (t) => { showLine('err', t); resetForm(); };
  const fd = new FormData();
  for (const [k,v] of new FormData(el('f'))) fd.append(k,v);
  const usePath = fd.get('local_path').trim() !== '';
  if (usePath) fd.delete('audio');
  el('status').innerText = usePath ? 'Starting…' : 'Uploading…';
  try {
    const r = await fetch('/dedup', {method:'POST', body: fd});
    const d = await r.json();
    if (d.error){ showErr(d.error); return; }
    const es = new EventSource('/progress/' + d.job_id);
    let drops = 0;
    es.onmessage = (ev) => {
      drops = 0;
      const m = JSON.parse(ev.data);
      el('bar').style.width = (m.progress*100)+'%';
      el('status').innerText = m.msg;
      if (m.done){
        es.close();
        if (m.error){ showErr(m.error); return; }
        showLine('done', '✅ Done: kept '+m.kept_sec+'s of '+m.total_sec+'s, removed about '+m.removed_sec+'s.');
        let resHtml = '<p><a href="/output/'+encodeURIComponent(m.outname)+'" download>⬇️ Download deduped audio</a></p><audio controls src="/output/'+encodeURIComponent(m.outname)+'"></audio>';
        el('res').innerHTML = resHtml;
        if (m.out_project) {
          const p = document.createElement('p');
          p.textContent = '🎬 Premiere project: ' + m.out_project;
          el('res').append(p);
        }
        if (m.out_cuts) {
          const p = document.createElement('p'), a = document.createElement('a');
          a.href = '/output/' + encodeURIComponent(m.out_cuts);
          a.target = '_blank';
          a.textContent = '📝 Cut list: read what was deleted';
          p.append(a);
          el('res').append(p);
        }
        for (const w of m.warnings) { const p = document.createElement('p'); p.className = 'warn'; p.textContent = '⚠️ '+w; el('res').append(p); }
        resetForm();
      }
    };
    // 浏览器会自动重连；连续失败说明服务已经关了，不要让页面一直停在进度条上
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED || ++drops >= 5) {
        es.close();
        showErr('Lost the connection to the server. It may have been closed, which also stops the job.');
      }
    };
  } catch(e){ showErr('Request failed: '+e); }
});
// 记住上次的设置，不用每次重填。API key 和文件路径不存
const SAVED_FIELDS = ['api_url','model','api_mode','language','whisper','keep_gap','out_format','out_bitrate'];
const form = document.getElementById('f');
const saved = JSON.parse(localStorage.getItem('settings') || '{}');
for (const k of SAVED_FIELDS) { if (saved[k]) form.elements[k].value = saved[k]; }
form.addEventListener('submit', () => localStorage.setItem('settings',
  JSON.stringify(Object.fromEntries(SAVED_FIELDS.map((k) => [k, form.elements[k].value])))));
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/dedup", methods=["POST"])
def dedup():
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
    if whisper not in ("large", "small"):
        whisper = "large"
    try:
        keep_gap = float(request.form.get("keep_gap") or 0.8)
    except ValueError:
        keep_gap = float("nan")
    if not 0.1 <= keep_gap <= 3:
        return {"error": "Silence gap must be a number between 0.1 and 3 seconds."}, 400
    out_format = request.form.get("out_format", "auto")
    out_bitrate = request.form.get("out_bitrate", "320k")
    language = request.form.get("language", "en").strip() or "en"
    if language not in ("en", "zh"):
        language = "en"
    outname = request.form.get("outname", "")

    if api_mode not in ("auto", "chat", "responses"):
        api_mode = "auto"
    if not (api_url and model and api_key):
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
                         "kept_sec": 0.0, "removed_sec": 0.0, "total_sec": 0.0, "outname": out_url,
                         "out_project": None, "out_cuts": None, "warnings": []}

    def _progress(frac, msg):
        with _jobs_lock:
            _jobs[job_id]["progress"] = frac
            _jobs[job_id]["msg"] = msg

    def worker():
        try:
            with _run_lock:
                info = dedup_audio(audio_path, out_path, api_url=api_url, model=model, api_key=api_key,
                                   whisper_model=whisper, language=language, keep_gap=keep_gap,
                                   api_mode=api_mode, out_format=out_format, out_bitrate=out_bitrate,
                                   progress=_progress, work_dir=job_dir)
            with _jobs_lock:
                j = _jobs[job_id]
                j["kept_sec"] = info["kept_sec"]
                j["removed_sec"] = info["removed_sec"]
                j["total_sec"] = info["total_sec"]
                j["msg"] = "Done"
                j["progress"] = 1.0
                j["out_project"] = info.get("out_project")
                j["out_cuts"] = os.path.basename(info["out_cuts"]) if info.get("out_cuts") else None
                j["warnings"] = info.get("warnings", [])
                j["done"] = True
        except Exception as e:
            traceback.print_exc()
            with _jobs_lock:
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["msg"] = "Error"
                _jobs[job_id]["done"] = True
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            _release_out_path(out_path)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.route("/progress/<job_id>")
def progress(job_id):
    def gen():
        last = None
        while True:
            # 在锁内取快照：否则可能读到 done=True 而结果字段还没写好的中间状态
            with _jobs_lock:
                j = _jobs.get(job_id)
                snapshot = dict(j) if j else None
            if snapshot is None:
                yield "data: " + json.dumps({"progress": 0.0, "msg": "Error", "done": True,
                                             "error": "Job not found. The server may have been restarted."}) + "\n\n"
                break
            data = json.dumps(snapshot)
            if data != last:
                last = data
                yield "data: " + data + "\n\n"
            if snapshot["done"]:
                break
            time.sleep(0.3)
    return Response(gen(), mimetype="text/event-stream")


@app.route("/output/<path:name>")
def out_file(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    # 先确认没有其他实例在运行，再清理 .work：否则再次启动会删掉运行中任务的文件
    claimed = _claim_port()
    if claimed is None:
        print(f"Port {PORT} is already in use. If Audio Dedup is already running, "
              f"open http://{HOST}:{PORT} in your browser.")
        sys.exit(1)
    _clean_stale_work()
    claimed.close()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
