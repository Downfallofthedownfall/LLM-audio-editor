"""
不需要显卡、LLM 和 Premiere 的测试。在项目目录运行:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import gzip, json, os, random, re, sys, tempfile, unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import dedup
import evaluate
import premiere


def _words(*rows):
    return [{"word": w, "start": float(s), "end": float(e)} for w, s, e in rows]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sentences(count, words_per_line=5):
    """count 句话，每句 words_per_line 个词，句与句之间紧挨着。"""
    words, t = [], 0.0
    for n in range(count):
        for k in range(words_per_line):
            text = "s%dw%d" % (n, k) + ("." if k == words_per_line - 1 else "")
            words.append({"word": text, "start": t, "end": t + 0.2})
            t += 0.25
    return words


class _Patched(unittest.TestCase):
    """把日志、缓存和各个阈值指到临时目录 / 测试用的值，结束后还原。"""
    patches = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        values = {"_LOG_FILE": os.path.join(self.tmp.name, "debug.log"),
                  "LLM_CACHE_DIR": os.path.join(self.tmp.name, "llm_cache"),
                  "EDIT_DIR": os.path.join(self.tmp.name, "edits"),
                  "_LLM_RETRY_WAIT_SEC": 0}
        values.update(self.patches)
        values["_dbg"] = lambda message: None
        for name, value in values.items():
            self.addCleanup(setattr, dedup, name, getattr(dedup, name))
            setattr(dedup, name, value)


class LinesTest(_Patched):
    def test_breaks_at_sentence_end_pause_and_cut(self):
        words = _words(("Hello", 0, .3), ("there.", .3, .6), ("So", .7, .9), ("we", 2.0, 2.2), ("go", 2.2, 2.4),
                       ("x-", 2.4, 2.5), ("now", 2.5, 2.7))
        self.assertEqual(dedup._lines(words, {5}), [[0, 1], [2], [3, 4], [6]])

    def test_silence_does_not_count_deleted_words(self):
        words = _words(("a.", 0, 1), ("um", 1.5, 2.5), ("b.", 6, 7))
        lines = dedup._lines(words, {1})
        self.assertEqual(lines, [[0], [2]])
        self.assertAlmostEqual(dedup._silence_before(words, lines, 1), 4.0)
        rows = dedup._line_rows(words, lines, 0, 2, " ", "en")
        self.assertEqual(rows, ["L0  a.", "L1  (after a 4 s pause) b."])

    def test_chinese_lines(self):
        words = _words(("今", 0, .2), ("天", .2, .4), ("好。", .4, .6), ("嗯", .7, .9), ("重", 4, 4.2), ("来", 4.2, 4.4))
        lines = dedup._lines(words, {3})
        self.assertEqual(lines, [[0, 1, 2], [4, 5]])
        self.assertEqual(dedup._line_rows(words, lines, 0, 2, dedup._word_sep("zh"), "zh"),
                         ["L0  今天好。", "L1  （停顿 3 秒后）重来"])
        self.assertIn("画外话", dedup._take_prompts("zh")[0])

    def test_long_line_breaks_at_comma_then_at_the_limit(self):
        words = [{"word": "w%d%s" % (i, "," if i == 34 else ""), "start": i * .2, "end": i * .2 + .2} for i in range(120)]
        self.assertEqual([len(line) for line in dedup._lines(words, set())], [35, 60, 25])

    def test_windows_cover_every_line_once(self):
        lines = dedup._lines(_sentences(12), set())
        with_context = dedup._line_windows(lines, 10)
        self.assertEqual([(a, b) for _, a, b, _ in with_context][0], (0, 2))
        judged = [n for _, a, b, _ in with_context for n in range(a, b)]
        self.assertEqual(judged, list(range(12)))


class ParseTakesTest(unittest.TestCase):
    def test_pairs_and_bare_numbers(self):
        off, pairs, unsure = dedup._parse_takes('{"off_script":[1,"2"],"earlier_take":[[3,5],[7],8,[9,10,11]],"unsure":[4]}')
        self.assertEqual((off, pairs, unsure), ({1, 2}, [(3, 5)], {4, 7, 8, 9}))

    def test_reasoning_before_the_answer(self):
        off, pairs, unsure = dedup._parse_takes('<think>[UM] {x}</think> Sure: {"off_script":[],"earlier_take":[[1,2]]}')
        self.assertEqual((off, pairs, unsure), (set(), [(1, 2)], set()))

    def test_wrong_shape_is_an_error(self):
        with self.assertRaises(ValueError):
            dedup._parse_takes('{"indices":[1,2]}')


class _LineLLM:
    """按行号作答的替身。answers(读的是第几遍, 待判定的行号) 返回回复的字典。"""
    working = ("chat", {"reasoning_effort": "high"})

    def __init__(self, answers, first_reading_windows):
        self.answers, self.first = answers, first_reading_windows
        self.calls = 0

    def complete(self, messages, timeout=None):
        self.calls += 1
        prompt = messages[-1]["content"]
        body = prompt.split("=== LINES TO JUDGE")[1].split("=== CONTEXT AFTER")[0]
        judged = tuple(int(n) for n in re.findall(r"^L(\d+)  ", body, flags=re.M))
        return json.dumps(self.answers(0 if judged in self.first else 1, judged))


class JudgeLinesTest(_Patched):
    # 12 行、每行 5 个词：第一遍的窗口是 0-3、4-7、8-11 行，第二遍是 0-1、2-5、6-9、10-11 行
    patches = {"_TAKE_WINDOW_WORDS": 20, "_TAKE_CONTEXT_WORDS": 10}
    FIRST = {(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)}

    def _run(self, answers, words=None):
        words = words or _sentences(12)
        lines = dedup._lines(words, set())
        llm = _LineLLM(answers, self.FIRST)
        cut, flag, failures, windows = dedup.llm_judge_lines(words, set(), llm)

        def _by_line(marks):
            return {n: marks[idxs[0]] for n, idxs in enumerate(lines) if idxs[0] in marks}
        return _by_line(cut), _by_line(flag), failures, windows

    def test_cut_needs_both_readings_and_a_take_needs_a_survivor(self):
        def answers(reading, judged):
            off = [n for n in judged if n in (1, 4, 6) or (n == 8 and reading == 0)]
            takes = [[9, 11]] if 9 in judged else []
            if 3 in judged:  # 两遍都说第 3 行是前一版，但取代它的行（4 和 6）都被剪掉了
                takes.append([3, 4] if reading == 0 else [3, 6])
            return {"off_script": off, "earlier_take": takes, "unsure": [n for n in judged if n == 7]}

        cut, flag, failures, windows = self._run(answers)
        self.assertEqual(failures, [])
        self.assertEqual(windows, 7)
        self.assertEqual(cut, {1: "off", 4: "off", 6: "off", 9: "retake"})
        self.assertEqual(flag, {3: "retake", 7: "unsure", 8: "off"})

    def test_take_replaced_by_an_off_script_line_in_the_same_window_is_only_marked(self):
        def answers(reading, judged):
            both = 9 in judged and 10 in judged
            return {"off_script": [10] if 10 in judged else [], "earlier_take": [[9, 10]] if 9 in judged else [],
                    "unsure": [] if both else []}

        cut, flag, _, _ = self._run(answers)
        self.assertEqual(cut, {10: "off"})
        self.assertEqual(flag, {9: "retake"})

    def test_replacement_must_be_a_later_line_that_was_shown(self):
        def answers(reading, judged):
            return {"off_script": [], "earlier_take": [[n, 0] for n in judged if n == 5] + [[n, 11] for n in judged if n == 2]}

        cut, flag, _, _ = self._run(answers)
        self.assertEqual(cut, {})
        self.assertEqual(flag, {2: "unsure", 5: "unsure"})

    def test_a_reply_that_cuts_most_of_a_window_is_refused(self):
        dedup._TAKE_WINDOW_WORDS, dedup._TAKE_CONTEXT_WORDS = 600, 300
        words = _sentences(30)  # 150 个词，只有一个窗口
        cut, flag, failures, windows = self._run(
            lambda reading, judged: {"off_script": list(judged)[:25], "earlier_take": []}, words)
        self.assertEqual((cut, flag), ({}, {}))
        self.assertEqual(len(failures), windows)

    def test_lines_outside_the_judged_section_are_ignored(self):
        cut, flag, _, _ = self._run(lambda reading, judged: {"off_script": list(range(12)), "earlier_take": []})
        self.assertEqual(sorted(cut), list(range(12)))  # 每个窗口只采用自己判定范围内的行，合起来才是全部
        self.assertEqual(flag, {})


class AskCacheTest(_Patched):
    class LLM:
        cache_id = "model-a"

        def __init__(self, replies):
            self.replies, self.calls = list(replies), 0

        def complete(self, messages, timeout=None):
            self.calls += 1
            return self.replies.pop(0)

    def test_parsed_reply_is_reused_and_a_refused_one_is_not_kept(self):
        messages = [{"role": "user", "content": "q"}]
        llm = self.LLM(["not json", '{"indices":[1]}'])
        parse = lambda content: json.loads(content)["indices"]
        with self.assertRaises(ValueError):
            dedup._ask(llm, messages, parse, "t")
        self.assertEqual(dedup._ask(llm, messages, parse, "t"), [1])
        self.assertEqual(dedup._ask(llm, messages, parse, "t"), [1])
        self.assertEqual(llm.calls, 2)
        other = self.LLM(['{"indices":[2]}'])
        other.cache_id = "model-b"
        self.assertEqual(dedup._ask(other, messages, parse, "t"), [2])

    def test_the_same_request_asked_twice_on_purpose_keeps_both_replies(self):
        messages = [{"role": "user", "content": "q"}]
        parse = lambda content: json.loads(content)["indices"]
        llm = self.LLM(['{"indices":[1]}', '{"indices":[2]}'])
        self.assertEqual([dedup._ask(llm, messages, parse, "t", ask=n) for n in (0, 1)], [[1], [2]])
        self.assertEqual([dedup._ask(llm, messages, parse, "t", ask=n) for n in (0, 1)], [[1], [2]])
        self.assertEqual(llm.calls, 2)

    def test_stand_in_without_cache_id_is_always_asked(self):
        llm = self.LLM(['{"indices":[1]}', '{"indices":[2]}'])
        del type(llm).cache_id
        try:
            parse = lambda content: json.loads(content)["indices"]
            self.assertEqual(dedup._ask(llm, [], parse, "t"), [1])
            self.assertEqual(dedup._ask(llm, [], parse, "t"), [2])
        finally:
            type(llm).cache_id = "model-a"


class CheckPointsTest(_Patched):
    def _tokens(self):
        words = _words(("Keep", 0, .4), ("this.", .5, .9), ("Oh", 2, 2.2), ("no.", 2.3, 2.6), ("uh", 2.7, 2.8),
                       ("Wrong", 2.9, 3.2), ("one.", 3.3, 3.6), ("Maybe", 5, 5.4), ("this.", 5.5, 5.9),
                       ("And", 6.1, 6.3), ("more.", 6.4, 6.8))
        takes = {2: "off", 3: "off", 5: "retake", 6: "retake"}
        return dedup._edit_tokens(words, {4}, {4}, takes, {7: "unsure", 8: "unsure"})

    def test_tokens_carry_cut_lock_and_flag(self):
        tokens = self._tokens()
        self.assertEqual([t["cut"] for t in tokens], [None, None, "off", "off", "first", "retake", "retake"] + [None] * 4)
        self.assertEqual([t["locked"] for t in tokens], [False, False, True, True, True, True, True] + [False] * 4)
        self.assertEqual([t.get("flag") for t in tokens][7:9], ["unsure", "unsure"])

    def test_points_merge_adjacent_lines_and_split_flags_at_gap_midpoints(self):
        points = dedup._check_points(self._tokens())
        self.assertEqual([(p["kind"], p["why"], p["text"]) for p in points],
                         [("cut", "off-script and earlier take", "Oh no. Wrong one."),
                          ("flag", "off-script or earlier take?", "Maybe this.")])
        self.assertEqual((points[0]["start"], points[0]["end"]), (2.0, 3.6))
        self.assertEqual((points[1]["start"], points[1]["end"]), (4.3, 6.0))

    def test_cut_list_marks_and_review_prompt_tag(self):
        tokens = self._tokens()
        keep = dedup.clean_keep_blocks(tokens, dedup._cut_indices(tokens), keep_gap=0.8)
        path = os.path.join(self.tmp.name, "cuts.txt")
        points = dedup._check_points(tokens)
        dedup.write_cut_list(tokens, keep, path, "en", (dedup._line_check_summary(points), ""), points)
        text = _read(path)
        self.assertIn("Line check: cut 1 passage (off-script remarks and earlier takes of something said again later); "
                      "kept and marked 1 passage it was less sure about.", text)
        self.assertIn("Check first (2 places;", text)
        self.assertIn("cut: off-script and earlier take", text)
        self.assertIn("[=Oh no.=]", text)
        self.assertIn("[-uh-] [=Wrong one.=]", text)
        self.assertIn("[?Maybe this.?]", text)
        rows = dedup._win_lines(tokens, 0, len(tokens), " ", dedup._cut_indices(tokens))
        self.assertIn("  2  OFF  Oh", rows)
        self.assertIn("  4  DEL  uh", rows)

    def test_take_mode_decides_what_is_cut(self):
        cut, flag = {1: "off", 2: "off", 5: "retake"}, {8: "unsure", 9: "retake"}
        self.assertEqual(dedup._apply_take_mode(cut, flag, "cut"),
                         ({1: "off", 2: "off"}, {8: "unsure", 9: "retake", 5: "sure_retake"}))
        self.assertEqual(dedup._apply_take_mode(cut, flag, "cut_takes"), (cut, flag))
        self.assertEqual(dedup._apply_take_mode(cut, flag, "mark"),
                         ({}, {8: "unsure", 9: "retake", 1: "sure_off", 2: "sure_off", 5: "sure_retake"}))
        self.assertEqual((cut, flag), ({1: "off", 2: "off", 5: "retake"}, {8: "unsure", 9: "retake"}))

    def test_default_mode_marks_an_agreed_earlier_take_without_a_question_mark(self):
        words = _words(("Keep", 0, .4), ("this.", .5, .9), ("Oh", 2, 2.2), ("no.", 2.3, 2.6), ("First", 3, 3.3),
                       ("try.", 3.4, 3.7), ("Maybe", 3.8, 4.1), ("this.", 4.2, 4.5), ("Second", 6, 6.3), ("try.", 6.4, 6.8))
        takes, flags = dedup._apply_take_mode({2: "off", 3: "off", 4: "retake", 5: "retake"}, {6: "retake", 7: "retake"}, "cut")
        tokens = dedup._edit_tokens(words, set(), set(), takes, flags)
        self.assertEqual([t["cut"] for t in tokens], [None, None, "off", "off"] + [None] * 6)
        points = dedup._check_points(tokens)
        self.assertEqual([(p["kind"], p["why"], p["text"]) for p in points],
                         [("cut", "off-script", "Oh no."), ("flag", "earlier take", "First try."),
                          ("flag", "earlier take?", "Maybe this.")])
        self.assertEqual(points[1]["end"], points[2]["start"])  # 两个片段首尾相接，把握不同所以分开
        self.assertEqual(dedup._line_check_summary(points),
                         "Line check: cut 1 passage (off-script remarks); kept and marked 1 passage that both readings "
                         "would cut (earlier takes of something said again later); kept and marked 1 passage it was "
                         "less sure about. They are listed at the top of the cut list and marked in the Premiere sequence.")
        keep = dedup.clean_keep_blocks(tokens, dedup._cut_indices(tokens), keep_gap=0.8)
        path = os.path.join(self.tmp.name, "cuts.txt")
        dedup.write_cut_list(tokens, keep, path, "en", (), points)
        text = _read(path)
        self.assertIn("kept: earlier take ", text)
        self.assertIn("kept: earlier take? ", text)
        for marked in ("[=Oh no.=]", "[?First try.?]", "[?Maybe this.?]"):
            self.assertIn(marked, text)

    def test_edit_record_round_trip(self):
        tokens = self._tokens()
        keep = dedup.clean_keep_blocks(tokens, dedup._cut_indices(tokens), keep_gap=0.8)
        path = dedup.save_edit(os.path.join(self.tmp.name, "out.flac"), "a.mp4", tokens, keep, 7.0, "en",
                               dedup._check_points(tokens))
        self.assertEqual(os.path.basename(path), "out.json")
        data = json.loads(_read(path))
        self.assertEqual([t.get("cut") for t in data["tokens"]][2:7], ["off", "off", "first", "retake", "retake"])
        self.assertEqual(data["tokens"][0], {"word": "Keep", "start": 0.0, "end": 0.4})
        self.assertEqual(len(data["keep"]), len(keep))


class HeardInvariantTest(_Patched):
    """听到的和计划的完全一样时，听成片不应该改动任何切点，存疑标记也要留着。"""

    def test_perfect_hearing_changes_nothing(self):
        rng = random.Random(7)
        words, t = [], 1.0
        for i in range(300):
            length = rng.uniform(0.18, 0.4)
            words.append({"word": "word%d%s" % (i, "." if i % 9 == 8 else ""), "start": t, "end": t + length})
            t += length + (rng.uniform(1.0, 3.0) if rng.random() < 0.08 else rng.uniform(0.06, 0.2))
        total = t + 1.0
        proposed = {i for i in range(300) if i % 17 in (3, 4) or i % 41 == 0}
        takes = {i: "off" for i in range(300) if 90 <= i < 99 or 180 <= i < 189}
        flags = {i: "retake" for i in range(300) if 207 <= i < 216}
        tokens = dedup._edit_tokens(words, proposed - set(takes), set(), takes, flags)
        env = np.full(int(total * dedup._ENV_RATE), 10.0, dtype=np.float32)
        for w in words:
            env[int(w["start"] * dedup._ENV_RATE):int(w["end"] * dedup._ENV_RATE)] = 60.0

        keep, timeline, word_block = dedup._plan_edit(tokens, 0.8, env, total)
        heard = [{"word": tokens[i]["word"], "start": timeline.to_edit(tokens[i]["start"]),
                  "end": timeline.to_edit(tokens[i]["end"])} for i in sorted(word_block)]
        merged, found = dedup._merge_heard(tokens, word_block, heard, timeline, env, "en")
        again, _, _ = dedup._plan_edit(merged, 0.8, env, total)

        self.assertEqual({k: v for k, v in found.items() if v}, {})
        self.assertEqual(len(again), len(keep))
        for (s0, e0), (s1, e1) in zip(keep, again):
            self.assertAlmostEqual(s0, s1, places=6)
            self.assertAlmostEqual(e0, e1, places=6)
        self.assertEqual([t.get("flag") for t in merged], [t.get("flag") for t in tokens])
        self.assertEqual([t.get("cut") for t in merged], [t.get("cut") for t in tokens])


class PremiereTest(unittest.TestCase):
    MEDIA = {"video": {"avg_frame_rate": "30/1", "width": 1920, "height": 1080},
             "audio": {"channels": 2, "sample_rate": "48000"}, "duration": 100.0, "audio_offset": 0.0}

    def setUp(self):
        self.addCleanup(setattr, premiere, "_probe_media", premiere._probe_media)
        premiere._probe_media = lambda path: dict(self.MEDIA)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_split_keeps_the_same_material(self):
        blocks = [(1.0, 5.0), (10.0, 20.0), (30.0, 31.0)]
        pieces = premiere._split_at_flags(blocks, [(4.95, 10.05), (12.0, 15.0), (19.9, 30.5), (30.5, 30.55)])
        self.assertEqual(pieces, [(1.0, 5.0, False), (10.0, 12.0, False), (12.0, 15.0, True), (15.0, 20.0, False),
                                  (30.0, 30.5, True), (30.5, 31.0, False)])
        self.assertAlmostEqual(sum(e - s for s, e, _ in pieces), sum(e - s for s, e in blocks))

    def test_flag_near_a_block_edge_takes_the_edge(self):
        self.assertEqual(premiere._split_at_flags([(10.0, 20.0)], [(10.1, 19.95)]), [(10.0, 20.0, True)])

    def test_xml_has_split_clips_labels_and_markers(self):
        path = os.path.join(self.tmp.name, "seq.xml")
        checks = [{"kind": "cut", "why": "off-script", "start": 6.0, "end": 8.0, "text": "oops <no> & why"},
                  {"kind": "flag", "why": "earlier take?", "start": 12.0, "end": 15.0, "text": "said twice"}]
        clips, markers = premiere.write_fcp_xml("a.mp4", [(1.0, 5.0), (10.0, 20.0)], path, "seq", checks)
        self.assertEqual(clips, 4)
        root = ET.parse(path).getroot()
        video = root.find("./sequence/media/video/track")
        spans = [(int(c.findtext("start")), int(c.findtext("end")), int(c.findtext("in")), int(c.findtext("out")),
                  c.findtext("./labels/label2")) for c in video.findall("clipitem")]
        self.assertEqual(spans, [(0, 120, 30, 150, None), (120, 180, 300, 360, None),
                                 (180, 270, 360, 450, premiere.FLAG_LABEL), (270, 420, 450, 600, None)])
        self.assertEqual(len(root.findall(".//clipitem")), 12)
        self.assertEqual(len(root.findall(".//labels")), 3)
        found = [(m.findtext("name"), m.findtext("comment"), int(m.findtext("in")), int(m.findtext("out")))
                 for m in root.findall("./sequence/marker")]
        self.assertEqual(found, [("Cut: off-script", "oops <no> & why", 120, -1),
                                 ("Check: earlier take?", "said twice", 180, 270)])
        self.assertEqual([(m["start_sec"], m["end_sec"]) for m in markers], [(4.0, -1.0), (6.0, 9.0)])

    def test_without_checks_the_sequence_is_unchanged(self):
        path = os.path.join(self.tmp.name, "seq.xml")
        clips, markers = premiere.write_fcp_xml("a.mp4", [(1.0, 5.0), (10.0, 20.0)], path, "seq")
        self.assertEqual((clips, markers), (2, []))
        text = _read(path)
        self.assertNotIn("<marker>", text)
        self.assertNotIn("<labels>", text)

    def test_import_script_gets_the_markers(self):
        script = premiere._IMPORT_SCRIPT % {"xml": '"a"', "project": '"b"', "status": '"c"',
                                            "markers": json.dumps([{"name": "Cut: 画外话", "comment": "x", "start_sec": 1.5,
                                                                    "end_sec": -1.0}])}
        self.assertIn('MARKERS = [{"name": "Cut: \\u753b\\u5916\\u8bdd"', script)


class EvaluateTest(unittest.TestCase):
    def _edit(self):
        tokens = [{"word": "w%d" % i, "start": float(i), "end": i + 0.8} for i in range(20)]
        for i in (3, 4, 10, 11, 12):
            tokens[i]["cut"] = "first"
        for i in (15, 16):
            tokens[i]["flag"] = "off"
        return {"source": "a.mp4", "language": "en", "tokens": tokens, "keep": [[0.0, 3.0], [5.0, 10.0], [13.0, 20.0]]}

    def test_counts_missed_wrong_and_edges(self):
        # 用户：恢复了第 10、11 个词，又删掉第 6、7 个和存疑的第 15、16 个，并把第一块的结尾延长了 0.3 秒
        truth = evaluate.merge([(0.0, 3.3), (5.0, 6.0), (8.0, 12.0), (13.0, 15.0), (17.0, 20.0)])
        result = evaluate.compare(self._edit(), truth)
        self.assertEqual((result["missed_words"], result["missed_places"], result["missed_flagged_words"]), (4, 2, 2))
        self.assertEqual((result["wrong_words"], result["wrong_places"]), (2, 1))
        self.assertEqual(result["wrong_by_reason"], {"first": 2})
        self.assertEqual(result["removed_by_both"], 3)
        self.assertAlmostEqual(result["found"], 3 / 7)
        self.assertAlmostEqual(result["missed_sec"], 4.0)
        self.assertAlmostEqual(result["wrong_sec"], 2.3)
        self.assertEqual(result["edges"]["tail_extended"], 1)
        text = evaluate.report(self._edit(), result, 0)
        self.assertIn("00:00:15 [marked]  w15 w16", text)
        self.assertIn("00:00:10 [first]  w10 w11", text)

    def test_reads_kept_ranges_from_a_project(self):
        ticks = evaluate.TICKS_PER_SECOND
        xml = ('<PremiereData Version="3">'
               '<AudioClipTrack ObjectUID="t"><ClipTrack><ClipItems><TrackItems>'
               '<TrackItem Index="0" ObjectRef="1"/><TrackItem Index="1" ObjectRef="4"/><TrackItem Index="2" ObjectRef="7"/>'
               '</TrackItems></ClipItems></ClipTrack></AudioClipTrack>'
               + "".join('<AudioClipTrackItem ObjectID="%d"><ClipTrackItem><SubClip ObjectRef="%d"/></ClipTrackItem></AudioClipTrackItem>'
                         '<SubClip ObjectID="%d"><Clip ObjectRef="%d"/><Name>%s</Name></SubClip>'
                         '<AudioClip ObjectID="%d"><Clip><InPoint>%d</InPoint><OutPoint>%d</OutPoint></Clip></AudioClip>'
                         % (n, n + 1, n + 1, n + 2, name, n + 2, start * ticks, end * ticks)
                         for n, name, start, end in ((1, "a.mp4", 5, 9), (4, "music.wav", 0, 60), (7, "a.mp4", 1, 2)))
               + '</PremiereData>')
        path = os.path.join(tempfile.mkdtemp(), "p.prproj")
        with gzip.open(path, "wb") as f:
            f.write(xml.encode("utf-8"))
        self.assertEqual(evaluate.read_project(path, "a.mp4"), [(1.0, 2.0), (5.0, 9.0)])


if __name__ == "__main__":
    unittest.main()
