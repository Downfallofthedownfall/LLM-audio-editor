#!/usr/bin/env python3
"""
对照评测：把工具的剪辑和人工在 Premiere 里修正过的工程逐词比较，算出漏删和误删。

人工修正后的工程就是标准答案：序列里留下的素材区间是该保留的，其余是该删的。
它按素材时间记录，所以同一份答案可以评测对同一个文件的任何一次运行（改了提示词或规则之后重跑再比）。

用法（在服务端的虚拟环境里运行，只用标准库）:
  python evaluate.py <判定记录.json> <修正后的.prproj 或 存下的答案.json> [--save-truth 答案.json] [--list N]

  判定记录   每次任务存在 .work\\edits\\<输出文件名>.json（见 dedup.save_edit）
  --save-truth  把从工程里读出的保留区间另存一份：工程以后被改动或删除，答案还在
  --list N      漏删和误删各列出最长的 N 处（默认 40，0 表示全部）

只统计两个词以上的差异时更可靠：单个词的差异多半是时间戳造成的（词的时间戳把它前面的停顿也算了进去）。
"""
import argparse, bisect, gzip, json, os, sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# Premiere 内部的时间单位：每秒的 tick 数
TICKS_PER_SECOND = 254016000000
# 切点挪动不超过这么多，算调整切点；更多算删改内容
EDGE_SEC = 0.5
# 切点挪动不到这么多不算挪动（工程里的切点取整到帧）
EDGE_TOLERANCE_SEC = 0.045

Interval = Tuple[float, float]


def merge(intervals: List[Interval], gap: float = 0.02) -> List[Interval]:
    """排序并合并相接的区间（相隔不超过 gap 秒的算相接：取整到帧之后相邻片段之间会差一点）。"""
    merged: List[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered(intervals: List[Interval], start: float, end: float) -> float:
    """[start, end] 落在 intervals（已排序、互不重叠）里的秒数。"""
    total = 0.0
    i = max(0, bisect.bisect_right(intervals, (start, float("inf"))) - 1)
    for a, b in intervals[i:]:
        if a >= end:
            break
        total += max(0.0, min(end, b) - max(start, a))
    return total


def subtract(intervals: List[Interval], minus: List[Interval]) -> List[Interval]:
    """intervals 里不在 minus 里的部分。"""
    out: List[Interval] = []
    for start, end in intervals:
        position = start
        for a, b in minus:
            if b <= position:
                continue
            if a >= end:
                break
            if a > position:
                out.append((position, a))
            position = max(position, b)
        if position < end:
            out.append((position, end))
    return out


def word_kept(intervals: List[Interval], word: Dict) -> bool:
    """这个词是否留在成片里：它的时间范围有一半以上落在保留区间里。"""
    start, end = float(word["start"]), float(word["end"])
    end = max(end, start + 0.02)
    return covered(intervals, start, end) >= 0.5 * (end - start)


def read_project(path: str, source_name: Optional[str] = None) -> List[Interval]:
    """读出 .prproj 序列里保留的素材区间（秒）。

    .prproj 是 gzip 压缩的 XML。轨道上的每个片段指向一个子剪辑，子剪辑指向带入点和出点的素材片段。
    工程里可能有好几条轨道和好几个序列：取引用这个素材（按文件名）时长最长的那一条音频轨，没有音频轨就取视频轨。
    """
    with gzip.open(path, "rb") as f:
        root = ET.parse(f).getroot()
    by_id = {el.get("ObjectID"): el for el in root if el.get("ObjectID")}

    def _ref(element, tag):
        child = element.find(tag) if element is not None else None
        return by_id.get(child.get("ObjectRef")) if child is not None else None

    best: List[Interval] = []
    for kind in ("AudioClipTrack", "VideoClipTrack"):
        for track in root.findall(kind):
            clips: List[Interval] = []
            for item in track.findall("./ClipTrack/ClipItems/TrackItems/TrackItem"):
                sub = _ref(by_id.get(item.get("ObjectRef")), "./ClipTrackItem/SubClip")
                clip = _ref(sub, "./Clip")
                inner = clip.find("./Clip") if clip is not None else None
                if inner is None or (source_name and (sub.findtext("Name") or "") != source_name):
                    continue
                clips.append((int(inner.findtext("InPoint") or 0) / TICKS_PER_SECOND,
                              int(inner.findtext("OutPoint") or 0) / TICKS_PER_SECOND))
            if sum(b - a for a, b in clips) > sum(b - a for a, b in best):
                best = clips
        if best:
            break
    if not best:
        raise RuntimeError("No clips of %s were found in %s" % (source_name or "any source", path))
    return merge(best)


def load_truth(path: str, source_name: Optional[str]) -> List[Interval]:
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return merge([(float(a), float(b)) for a, b in json.load(f)["keep"]])
    return read_project(path, source_name)


def _runs(indices: List[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def compare(edit: Dict, truth: List[Interval]) -> Dict:
    """逐词比较，返回统计和差异列表。

    工具留没留一个词，看它的保留区间（和答案用同一种量法），不看这个词的删除标记：
    这样两边的时间戳误差互相抵消。没存保留区间的旧记录才退回看标记。
    """
    tokens = edit["tokens"]
    tool = merge([(float(a), float(b)) for a, b in edit.get("keep") or []], gap=0.0)
    if tool:
        tool_keeps = [not t.get("cut") and word_kept(tool, t) for t in tokens]
    else:
        tool_keeps = [not t.get("cut") for t in tokens]
    user_keeps = [word_kept(truth, t) for t in tokens]

    missed = [i for i, t in enumerate(tokens) if tool_keeps[i] and not user_keeps[i]]
    wrong = [i for i, t in enumerate(tokens) if not tool_keeps[i] and user_keeps[i]]
    both_removed = sum(1 for i in range(len(tokens)) if not tool_keeps[i] and not user_keeps[i])
    missed_runs = [r for r in _runs(missed) if len(r) >= 2]
    wrong_runs = [r for r in _runs(wrong) if len(r) >= 2]
    missed_words = sum(len(r) for r in missed_runs)
    wrong_words = sum(len(r) for r in wrong_runs)
    flagged = sum(1 for r in missed_runs for i in r if tokens[i].get("flag"))

    result = {
        "words": len(tokens),
        "kept_by_both": sum(1 for i in range(len(tokens)) if tool_keeps[i] and user_keeps[i]),
        "removed_by_both": both_removed,
        "missed_words": missed_words, "missed_places": len(missed_runs), "missed_flagged_words": flagged,
        "wrong_words": wrong_words, "wrong_places": len(wrong_runs),
        "single_word_differences": len(missed) + len(wrong) - missed_words - wrong_words,
        # 该删的词里工具找到了多少；工具删的词里有多少是对的
        "found": both_removed / max(1, both_removed + missed_words),
        "right": both_removed / max(1, both_removed + wrong_words),
        "missed_runs": missed_runs, "wrong_runs": wrong_runs,
    }
    by_cut: Dict[str, int] = {}
    for r in wrong_runs:
        for i in r:
            reason = tokens[i].get("cut") or "cut point"
            by_cut[reason] = by_cut.get(reason, 0) + 1
    result["wrong_by_reason"] = by_cut
    if tool:
        result["tool_sec"] = sum(b - a for a, b in tool)
        result["truth_sec"] = sum(b - a for a, b in truth)
        result["missed_sec"] = sum(b - a for a, b in subtract(tool, truth))
        result["wrong_sec"] = sum(b - a for a, b in subtract(truth, tool))
        result["edges"] = _edges(tool, truth)
    return result


def _edges(tool: List[Interval], truth: List[Interval]) -> Dict[str, int]:
    """工具的每个保留块，用户把它的开头、结尾挪了多少。只数挪动不超过 EDGE_SEC 的：那是在调切点，不是删改内容。"""
    counts = {"blocks": len(tool), "removed": 0, "untouched": 0,
              "head_trimmed": 0, "head_extended": 0, "tail_trimmed": 0, "tail_extended": 0}
    starts = [a for a, _ in truth]
    for n, (start, end) in enumerate(tool):
        inside = [(max(start, a), min(end, b)) for a, b in truth if min(end, b) > max(start, a)]
        if not inside:
            counts["removed"] += 1
            continue
        first, last = inside[0][0], inside[-1][1]
        # 用户的片段从哪里开始 / 到哪里结束（可能超出工具的块）；超出到相邻块里的不算延长
        user_start = truth[max(0, bisect.bisect_right(starts, first + 1e-6) - 1)][0]
        user_end = truth[max(0, bisect.bisect_right(starts, last - 1e-6) - 1)][1]
        prev_end = tool[n - 1][1] if n else float("-inf")
        next_start = tool[n + 1][0] if n + 1 < len(tool) else float("inf")
        head = first - start if first - start > EDGE_TOLERANCE_SEC else -(start - max(user_start, prev_end))
        tail = end - last if end - last > EDGE_TOLERANCE_SEC else -(min(user_end, next_start) - end)
        moved = False
        for amount, trimmed, extended in ((head, "head_trimmed", "head_extended"), (tail, "tail_trimmed", "tail_extended")):
            if EDGE_TOLERANCE_SEC < abs(amount) <= EDGE_SEC:
                counts[trimmed if amount > 0 else extended] += 1
                moved = True
            moved |= abs(amount) > EDGE_SEC
        counts["untouched"] += not moved and len(inside) == 1
    return counts


def _timecode(sec: float) -> str:
    s = int(sec)
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def report(edit: Dict, result: Dict, limit: int) -> str:
    tokens = edit["tokens"]
    sep = "" if edit.get("language") == "zh" else " "
    lines = []
    out = lines.append
    out("Source: %s" % edit.get("source", "?"))
    out("Words: %d. Kept by both: %d. Removed by both: %d." % (result["words"], result["kept_by_both"], result["removed_by_both"]))
    out("")
    out("MISSED  (the tool kept it, you removed it):  %d words in %d places" % (result["missed_words"], result["missed_places"]))
    if result["missed_words"]:
        out("        of these, the tool had marked %d words as 'check this' (%.0f%%)"
            % (result["missed_flagged_words"], 100 * result["missed_flagged_words"] / result["missed_words"]))
    out("WRONG   (the tool removed it, you kept it):  %d words in %d places" % (result["wrong_words"], result["wrong_places"]))
    if result["wrong_by_reason"]:
        out("        by reason: " + ", ".join("%s %d" % kv for kv in sorted(result["wrong_by_reason"].items())))
    out("Differences of a single word (mostly timestamp noise, not counted): %d" % result["single_word_differences"])
    out("")
    out("The tool found %.0f%% of the words that had to go, and %.1f%% of the words it removed were right to remove."
        % (100 * result["found"], 100 * result["right"]))
    if "tool_sec" in result:
        out("Length: the tool's edit %.0f s, yours %.0f s. You removed %.0f s more and put back %.0f s."
            % (result["tool_sec"], result["truth_sec"], result["missed_sec"], result["wrong_sec"]))
        e = result["edges"]
        out("Cut points: of the tool's %d blocks you left %d untouched and removed %d whole. Moved by up to %.1f s: "
            "%d starts trimmed, %d starts extended, %d ends trimmed, %d ends extended."
            % (e["blocks"], e["untouched"], e["removed"], EDGE_SEC, e["head_trimmed"], e["head_extended"],
               e["tail_trimmed"], e["tail_extended"]))

    def _text(indices: List[int], width: int = 220) -> str:
        text = sep.join(tokens[i]["word"].strip() for i in indices)
        return text if len(text) <= width else text[:width] + "…"

    for title, runs in (("MISSED", result["missed_runs"]), ("WRONG", result["wrong_runs"])):
        shown = sorted(runs, key=len, reverse=True)[:limit] if limit else runs
        out("")
        out("%s, %s (time in the source):" % (title, "all %d places" % len(runs) if len(shown) == len(runs)
                                               else "the %d longest of %d places" % (len(shown), len(runs))))
        for run in sorted(shown):
            mark = ""
            if title == "MISSED" and any(tokens[i].get("flag") for i in run):
                mark = " [marked]"
            if title == "WRONG":
                mark = " [%s]" % (tokens[run[0]].get("cut") or "cut point")
            out("  %s%s  %s" % (_timecode(float(tokens[run[0]]["start"])), mark, _text(run)))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare the tool's edit with a Premiere project you corrected by hand.")
    ap.add_argument("edit", help="the edit record of a run: .work\\edits\\<output name>.json")
    ap.add_argument("truth", help="the corrected .prproj, or a truth file saved with --save-truth")
    ap.add_argument("--save-truth", metavar="PATH", help="save the kept source ranges read from the project")
    ap.add_argument("--list", type=int, default=40, metavar="N", help="list the N longest differences of each kind (0 = all)")
    args = ap.parse_args(argv)
    with open(args.edit, encoding="utf-8") as f:
        edit = json.load(f)
    truth = load_truth(args.truth, os.path.basename(edit["source"]) if edit.get("source") else None)
    if args.save_truth:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_truth)), exist_ok=True)
        with open(args.save_truth, "w", encoding="utf-8") as f:
            json.dump({"source": edit.get("source"), "from": os.path.abspath(args.truth),
                       "keep": [[round(a, 3), round(b, 3)] for a, b in truth]}, f)
    text = report(edit, compare(edit, truth), args.list)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
