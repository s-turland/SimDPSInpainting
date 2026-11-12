import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf
from test_inpainter import SR, toy_prior

from simdps import Inpainter
from simdps.app import State, serve


@pytest.fixture
def server(tmp_path):
    srv = serve(State(Inpainter(toy_prior(), buffer=4), "toy", bank=str(tmp_path / "bank")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def _call(url, body=None, raw=False):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data)) as r:
        out = r.read()
    return out if raw else json.loads(out)


def _wav(seconds, channels=2):
    buf = io.BytesIO()
    x = np.random.default_rng(0).standard_normal((seconds * SR, channels)).astype(np.float32) * 0.1
    sf.write(buf, x, SR, format="WAV")
    return buf.getvalue()


def test_info(server):
    info = _call(server + "/api/info")
    assert info["ready"] and info["model"] == "toy" and info["sample_rate"] == SR


@pytest.mark.parametrize("match", ["latent", "audio", "bands"])
def test_search(server, match):
    track = _call(server + "/api/audio?name=take.wav", _wav(15))["id"]
    found = _call(server + "/api/search", {"track": track, "gap": [7.0, 7.05], "match": match, "n": 4})
    assert len(found["candidates"]) == 4


def test_inpaint_job(server):
    track = _call(server + "/api/audio?name=take.wav", _wav(3))["id"]
    other = _call(server + "/api/audio?name=other.wav", _wav(2, 1))["id"]
    guides = [{"source": track, "start": 0.4, "weight": 0.7}, {"source": other, "start": 0.9, "weight": 0.3}]
    job = _call(server + "/api/inpaint", {"track": track, "gap": [1.5, 1.55], "guides": guides,
                                         "samples": 2, "preset": "warm_start_fast", "seed": 1})["id"]
    for _ in range(200):
        st = _call(server + f"/api/job/{job}")
        if st["done"]:
            break
        time.sleep(0.05)
    assert st["error"] is None and st["step"] == st["steps"] and len(st["results"]) == 2
    y, sr = sf.read(io.BytesIO(_call(server + f"/api/result/{st['results'][0]}", raw=True)))
    assert sr == SR and y.shape == (3 * SR, 2)


def test_errors(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _call(server + "/api/audio?name=x.wav", b"not audio")
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError):
        _call(server + "/api/inpaint", {"track": "nope", "gap": [1, 2], "guides": []})


def test_serves_page(server):
    assert b"<canvas" in _call(server + "/", raw=True)
    assert b"class LCD" in _call(server + "/lcd.js", raw=True)
    with pytest.raises(urllib.error.HTTPError):
        _call(server + "/../app.py", raw=True)


def _wait(server, job):
    for _ in range(400):
        st = _call(server + f"/api/job/{job}")
        if st["done"]:
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_checkpoints_and_restart(server):
    track = _call(server + "/api/audio?name=take.wav", _wav(3))["id"]
    guides = [{"source": track, "start": 0.4, "weight": 0.3}, {"source": track, "start": 2.2, "weight": 0.7},
              {"source": track, "start": 0.9, "weight": 0.0}]
    job = _call(server + "/api/inpaint", {"track": track, "gap": [1.5, 1.55], "guides": guides, "samples": 2,
                                         "preset": "warm_start_detailed", "seed": 1, "checkpoints": 4})["id"]
    st = _wait(server, job)
    assert st["error"] is None and len(st["checkpoints"]) == 4
    tr = st["trace"]
    assert len(tr["guides"]) == 3 and len(tr["paths"]) == 2 and len(tr["paths"][0]) == st["steps"]
    y, sr = sf.read(io.BytesIO(_call(server + f"/api/checkpoint/{job}/2/1", raw=True)))
    assert y.shape == (3 * SR, 2)

    c = st["checkpoints"][1]
    restart = {"restart": {"job": job, "index": 1, "sample": 1}, "samples": 3, "seed": 5}
    again = _call(server + "/api/inpaint", restart)["id"]
    st2 = _wait(server, again)
    assert st2["error"] is None and len(st2["results"]) == 3 and st2["first"] == c["step"]
    assert st2["parent"]["job"] == job and len(st2["trace"]["paths"]) == 3
    # a variation carries on from its parent's path, laid out afresh
    assert [p[0] for p in st2["trace"]["paths"][0][: c["step"]]] == list(range(1, c["step"] + 1))
    assert len(st2["trace"]["paths"][0]) == st2["steps"]


def test_bank(server):
    assert _call(server + "/api/bank")["files"] == []
    f = _call(server + "/api/bank?name=my%20fill.wav&kind=fill&origin=take.wav&at=0.5&length=0.05", _wav(2))["file"]
    assert f == "my_fill_01.wav"
    e = _call(server + "/api/bank")["files"][0]
    assert e["kind"] == "fill" and e["at"] == 0.5 and e["dur"] == 2 and e["id"] == "bank:" + f
    assert sf.read(io.BytesIO(_call(server + "/api/bank/" + f, raw=True)))[0].shape == (2 * SR, 2)

    track = _call(server + "/api/audio?name=take.wav", _wav(15))["id"]
    found = _call(server + "/api/search", {"track": track, "gap": [7.0, 7.05], "source": "bank", "n": 2})["candidates"]
    assert found and all(c["source"] == "bank:" + f for c in found)
    guides = [{"source": "bank:" + f, "start": found[0]["start"], "weight": 1.0}]
    job = _call(server + "/api/inpaint", {"track": track, "gap": [7.0, 7.05], "guides": guides, "seed": 0})["id"]
    assert _wait(server, job)["error"] is None

    _call(server + f"/api/bank/{f}/drop", b"")
    assert _call(server + "/api/bank")["files"] == []
    with pytest.raises(urllib.error.HTTPError):
        _call(server + "/api/bank/..%2Fbank.json", raw=True)
    with pytest.raises(urllib.error.HTTPError):
        _call(server + "/api/bank?name=x.wav", b"not audio")


def test_restart_with_new_guides(server):
    track = _call(server + "/api/audio?name=take.wav", _wav(3))["id"]
    guides = [{"source": track, "start": 0.4, "weight": 1.0}]
    job = _call(server + "/api/inpaint", {"track": track, "gap": [1.5, 1.55], "guides": guides,
                                         "preset": "warm_start_detailed", "seed": 1, "checkpoints": 3})["id"]
    st = _wait(server, job)
    other = [{"source": track, "start": 2.2, "weight": 0.6}, {"source": track, "start": 0.9, "weight": 0.4}]
    again = _call(server + "/api/inpaint", {"restart": {"job": job, "index": 0}, "guides": other,
                                           "seed": 2})["id"]
    st2 = _wait(server, again)
    first = st["checkpoints"][0]["step"]
    assert st2["error"] is None and not st2["parent"]["same_guides"]
    assert len(st2["trace"]["guides"]) == 2 and st2["trace"]["paths"][0][0][0] == first + 1
    # checkpoints after the restart only; the earlier ones stay with the first job
    assert [c["step"] for c in st2["checkpoints"]] == [c["step"] for c in st["checkpoints"][1:]]


def test_paper_settings_by_default(server):
    track = _call(server + "/api/audio?name=take.wav", _wav(3))["id"]
    guides = [{"start": 0.5}]
    job = _call(server + "/api/inpaint", {"track": track, "gap": [1.5, 1.55], "guides": guides, "seed": 1})["id"]
    assert _wait(server, job)["steps"] == 100


def test_settings_of_a_run(server):
    s = _call(server + "/api/settings?preset=warm_start_fast&method=solo")
    assert s["method"] == "solo" and s["sampler"]["steps"] == 35 and s["sampler"]["sigma_max"] == 1
    assert s["candidates"]["combine"] == "solo" and s["guidance"]["jacobian"] is False
    assert _call(server + "/api/settings")["candidates"]["combine"] == "weighted"


def test_custom_settings(server):
    track = _call(server + "/api/audio?name=take.wav", _wav(3))["id"]
    body = {"track": track, "gap": [1.5, 1.55], "guides": [{"start": 0.5}], "seed": 1,
            "method": "multi", "settings": {"sampler": {"steps": 7}, "candidates": {"taper": None}}}
    assert _wait(server, _call(server + "/api/inpaint", body)["id"])["steps"] == 7
    for bad in ({"sampler": {"nope": 1}}, {"nope": {}}, {"sampler": {"steps": [1]}}):
        with pytest.raises(urllib.error.HTTPError):
            _call(server + "/api/inpaint", {**body, "settings": bad})


def test_unknown_preset_is_refused(server):
    track = _call(server + "/api/audio?name=take.wav", _wav(3))["id"]
    guides = [{"source": track, "start": 0.4, "weight": 1.0}]
    with pytest.raises(urllib.error.HTTPError):
        _call(server + "/api/inpaint", {"track": track, "gap": [1.5, 1.55], "guides": guides, "preset": "full"})


@pytest.mark.parametrize("headers", [{"Origin": "https://example.com"}, {"Host": "example.com"}])
def test_refuses_other_sites(server, headers):
    for data in (None, b"{}"):
        req = urllib.request.Request(server + "/api/info", data=data, headers=headers)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403


def test_accepts_its_own_page(server):
    req = urllib.request.Request(server + "/api/search", data=b"{}", headers={"Origin": server})
    with pytest.raises(urllib.error.HTTPError) as e:  # reaches the handler, which wants a track
        urllib.request.urlopen(req)
    assert e.value.code == 400


def test_single_guide_uses_solo(tmp_path):
    inp, seen = Inpainter(toy_prior(), buffer=4), []
    inpaint = inp.inpaint
    inp.inpaint = lambda *a, **k: seen.append(k["method"]) or inpaint(*a, **k)
    srv = serve(State(inp, "toy", bank=str(tmp_path / "bank")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        track = _call(url + "/api/audio?name=take.wav", _wav(3))["id"]
        for guides in ([{"start": 0.4}], [{"start": 0.4}, {"start": 0.9}]):
            job = _call(url + "/api/inpaint", {"track": track, "gap": [1.5, 1.55], "guides": guides,
                                               "preset": "warm_start_fast", "seed": 1})["id"]
            assert _wait(url, job)["error"] is None
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen == ["solo", None]
