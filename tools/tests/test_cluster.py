import os
import types

import pytest
from click.testing import CliRunner

from sparkyard import cli, cluster
from sparkyard.render import load

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

IBDEV = """rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
"""


KEYSCAN_BOTH = "10.0.0.2 ssh-ed25519 AAA\n192.168.100.11 ssh-ed25519 AAA\n"


def _settings(tmp_path):
    """Cluster fixture settings with repo_path/llm_root redirected under tmp_path."""
    import yaml
    d = yaml.safe_load(open(os.path.join(FIX, "settings.cluster.yaml")))
    d["repo_path"] = str(tmp_path / "repo")
    d["llm_root"] = str(tmp_path / "llm")
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(d))
    s, models, _ = load(os.path.join(FIX, "models.cluster.yaml"), str(p))
    return s, models, p


def _cp(rc=0, out="", err=""):
    return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)


class Fake:
    """argv-dispatching fake `run`; `rules` map a predicate on the joined argv
    to a result. Records every argv."""
    def __init__(self, rules=None, default=None):
        self.calls, self.rules = [], rules or []
        self.default = default or _cp()

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        j = " ".join(argv)
        for pred, res in self.rules:
            if pred in j:
                return res(argv) if callable(res) else res
        return self.default


def _healthy_rules(img_head="sha256:aa", img_worker="sha256:aa", worker_sizes="100\n50\n"):
    def inspect(argv):
        return _cp(0, img_worker if argv[0] == "ssh" else img_head)
    return [
        ("docker info --format {{.ServerVersion}}", _cp(0, "27.0")),
        ("{{json .Runtimes}}", _cp(0, '{"nvidia":{},"runc":{}}')),
        ("docker image inspect", inspect),
        ("operstate", _cp(0, "up")),
        ("ports/1/state", _cp(0, "4: ACTIVE")),
        ("ping", _cp(0)),
        ("-printf", _cp(0, worker_sizes)),
    ]


def _make_weights(s, sizes=(100, 50)):
    d = os.path.join(s.llm_root, "vllm/Org/Big")
    os.makedirs(d)
    for i, n in enumerate(sizes):
        with open(os.path.join(d, f"m{i}.safetensors"), "wb") as f:
            f.write(b"x" * n)


# --- setup-key ---

def test_setup_key_prints_copy_id_and_never_touches_authorized_keys(tmp_path, capsys):
    s, _, _ = _settings(tmp_path)
    run = Fake([("ssh-keyscan", _cp(0, KEYSCAN_BOTH))])
    assert cluster.setup_key(s, run=run) == 0
    assert not any("authorized_keys" in a for argv in run.calls for a in argv)
    d = f"{tmp_path}/repo/llama-swap/cluster"
    assert f"ssh-copy-id -i {d}/id_ed25519.pub acme@10.0.0.2" in capsys.readouterr().out
    assert os.stat(d).st_mode & 0o777 == 0o700
    assert open(f"{d}/known_hosts").read().startswith("10.0.0.2")
    keygen = [c for c in run.calls if c[0] == "ssh-keygen" and "-t" in c][0]
    assert keygen == ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "sparkyard-llama-swap",
                      "-f", f"{d}/id_ed25519"]
    assert ["ssh-keyscan", "-t", "ed25519", "10.0.0.2", "192.168.100.11"] in run.calls


@pytest.mark.parametrize("scan", ["10.0.0.2 ssh-ed25519 AAA\n",
                                  "192.168.100.11 ssh-ed25519 AAA\n",
                                  "# 10.0.0.2:22 SSH-2.0-OpenSSH\n192.168.100.11 ssh-ed25519 AAA\n"])
def test_setup_key_requires_keys_for_both_hosts(tmp_path, capsys, scan):
    s, _, _ = _settings(tmp_path)
    run = Fake([("ssh-keyscan", _cp(0, scan))])
    assert cluster.setup_key(s, run=run) == 1
    out = capsys.readouterr().out
    assert "✗" in out and "ssh-keyscan" in out
    assert not os.path.exists(os.path.join(s.cluster_dir, "known_hosts"))


def test_setup_key_accepts_hashed_or_bracketed_host_fields(tmp_path):
    s, _, _ = _settings(tmp_path)
    scan = "10.0.0.2,spark-2 ssh-ed25519 AAA\n[192.168.100.11]:22 ssh-ed25519 AAA\n"
    assert cluster.setup_key(s, run=Fake([("ssh-keyscan", _cp(0, scan))])) == 0


def test_setup_key_prints_from_restriction_hint(tmp_path, capsys):
    s, _, _ = _settings(tmp_path)
    assert cluster.setup_key(s, run=Fake([("ssh-keyscan", _cp(0, KEYSCAN_BOTH))])) == 0
    out = capsys.readouterr().out
    assert 'from="<head LAN IP>,192.168.100.10"' in out
    assert "authorized_keys" in out


def test_setup_key_keeps_existing_key(tmp_path):
    s, _, _ = _settings(tmp_path)
    os.makedirs(s.cluster_dir)
    open(os.path.join(s.cluster_dir, "id_ed25519"), "w").write("k")
    run = Fake([("ssh-keyscan", _cp(0, KEYSCAN_BOTH))])
    assert cluster.setup_key(s, run=run) == 0
    assert not any(c[0] == "ssh-keygen" and "-t" in c for c in run.calls)


def test_ssh_base_matches_launch_options(tmp_path):
    s, _, _ = _settings(tmp_path)
    d = s.cluster_dir
    assert cluster.ssh_base(s, "10.0.0.2") == [
        "ssh", "-i", f"{d}/id_ed25519", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        "-o", f"UserKnownHostsFile={d}/known_hosts", "-o", "StrictHostKeyChecking=yes",
        "-l", "acme", "10.0.0.2"]


# --- check ---

def test_check_all_pass_returns_0(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    assert cluster.check(s, models, run=Fake(_healthy_rules())) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    for label in ("ssh", "ssh cx7", "worker rsync", "worker docker", "worker nvidia runtime", "image vllm-node:latest",
                  "iface enp1s0f0np0 head", "iface enp1s0f0np0 worker",
                  "hca rocep1s0f0 head", "hca roceP2p1s0f0 worker", "cx7 ping",
                  "weights Big-TP2"):
        assert f"PASS  {label}" in out, label


def test_check_image_mismatch_fails(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    assert cluster.check(s, models, run=Fake(_healthy_rules(img_worker="sha256:bb"))) == 1
    assert "FAIL  image vllm-node:latest" in capsys.readouterr().out


def test_check_weights_size_mismatch_fails(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    assert cluster.check(s, models, run=Fake(_healthy_rules(worker_sizes="100\n"))) == 1
    assert "FAIL  weights Big-TP2" in capsys.readouterr().out


def test_check_nvidia_runtime_missing_shows_fix_hint(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    rules = [("{{json .Runtimes}}", _cp(0, '{"runc":{}}'))] + _healthy_rules()
    assert cluster.check(s, models, run=Fake(rules)) == 1
    out = capsys.readouterr().out
    assert ("FAIL  worker nvidia runtime  sudo nvidia-ctk runtime configure "
            "--runtime=docker --set-as-default && sudo systemctl restart docker") in out


def test_check_ssh_cx7_and_rsync_lines(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    run = Fake(_healthy_rules())
    assert cluster.check(s, models, run=run) == 0
    out = capsys.readouterr().out
    assert "PASS  ssh cx7  acme@192.168.100.11" in out
    cx = [c for c in run.calls if c[0] == "ssh" and c[-3:] == ["192.168.100.11", "--", "true"]]
    assert cx, run.calls
    rs = [c for c in run.calls if c[0] == "ssh" and c[-1] == "command -v rsync"]
    assert rs and rs[0][-3] == "10.0.0.2"


def test_check_ssh_cx7_failure_reported(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    rules = [("192.168.100.11 -- true", _cp(255, "", "Host key verification failed."))] \
        + _healthy_rules()
    assert cluster.check(s, models, run=Fake(rules)) == 1
    out = capsys.readouterr().out
    assert "FAIL  ssh cx7" in out and "Host key verification failed." in out


def test_check_worker_rsync_missing(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    rules = [("command -v rsync", _cp(1))] + _healthy_rules()
    assert cluster.check(s, models, run=Fake(rules)) == 1
    assert "FAIL  worker rsync" in capsys.readouterr().out


def test_check_ssh_failure_stops_early(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    run = Fake([("-- true", _cp(255, "", "Permission denied"))])
    assert cluster.check(s, models, run=run) == 1
    out = capsys.readouterr().out
    assert out.count("FAIL") == 1 and "FAIL  ssh" in out
    assert len(run.calls) == 1


# --- suggest ---

def test_suggest_parses_ibdev2netdev():
    out = cluster.suggest(run=Fake(default=_cp(0, IBDEV)))
    assert "eth_if: enp1s0f0np0" in out
    assert "ib_hca: rocep1s0f0,roceP2p1s0f0" in out
    assert all(ln.startswith("#") for ln in out.splitlines())


def test_suggest_without_ibdev2netdev():
    def run(argv, **kw):
        raise FileNotFoundError("ibdev2netdev")
    assert "two-node-vllm-design" in cluster.suggest(run=run)


# --- sync ---

class FakePopen:
    instances = []

    def __init__(self, argv, stdout=None, stdin=None, rc=0):
        self.argv, self.stdin, self.rc = argv, stdin, rc
        self.stdout = types.SimpleNamespace(closed=False, close=self._close) if stdout else None
        self.waited = False
        FakePopen.instances.append(self)

    def _close(self):
        self.stdout.closed = True

    def wait(self):
        self.waited = True
        return self.rc


def _popen(rcs=(0, 0)):
    FakePopen.instances = []
    it = iter(rcs)
    return lambda argv, **kw: FakePopen(argv, rc=next(it), **kw)


def test_sync_skips_image_when_ids_match(tmp_path):
    s, models, _ = _settings(tmp_path)
    run = Fake(_healthy_rules())
    popen = _popen()
    assert cluster.sync(s, models, [], run=run, popen=popen) == 0
    assert FakePopen.instances == []


def test_sync_streams_image_when_ids_differ(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    run = Fake(_healthy_rules(img_worker="sha256:bb"))
    assert cluster.sync(s, models, [], run=run, popen=_popen()) == 0
    save, load = FakePopen.instances
    assert save.argv == ["docker", "save", "vllm-node:latest"]
    assert load.argv[0] == "ssh" and load.argv[-2:] == ["--", "docker load"]
    assert load.argv[-3] == "192.168.100.11"   # over the CX7 link, not ssh_host
    assert load.stdin is save.stdout and save.stdout.closed
    assert save.waited and load.waited


@pytest.mark.parametrize("rcs", [(0, 1), (1, 0)])
def test_sync_image_stream_failure_returns_1(tmp_path, capsys, rcs):
    s, models, _ = _settings(tmp_path)
    run = Fake(_healthy_rules(img_worker=""))
    assert cluster.sync(s, models, [], run=run, popen=_popen(rcs)) == 1
    assert "✗" in capsys.readouterr().out


def test_sync_rsyncs_over_cx7_ip(tmp_path):
    s, models, _ = _settings(tmp_path)
    run = Fake(_healthy_rules())
    assert cluster.sync(s, models, ["Big-TP2"], run=run, popen=_popen()) == 0
    src = f"{tmp_path}/llm/vllm/Org/Big/"
    rs = [c for c in run.calls if c[0] == "rsync"][0]
    assert rs[:3] == ["rsync", "-a", "--info=progress2"]
    assert rs[-2:] == [src, f"acme@192.168.100.11:{src}"]
    e = rs[rs.index("-e") + 1]
    assert "StrictHostKeyChecking=yes" in e and e.endswith("-l acme")
    assert "192.168.100.11" not in e
    mk = [c for c in run.calls if c[0] == "ssh" and "mkdir" in " ".join(c)][0]
    assert mk[-1] == f"mkdir -p {tmp_path}/llm/vllm/Org"


def test_sync_unknown_model(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    run = Fake(_healthy_rules())
    assert cluster.sync(s, models, ["nope"], run=run, popen=_popen()) == 1
    assert "✗ unknown model" in capsys.readouterr().out
    assert run.calls == []


# --- CLI wiring ---

def test_cli_cluster_help_lists_subcommands():
    out = CliRunner().invoke(cli.cli, ["cluster", "--help"]).output
    cmds = out[out.index("Commands:"):]
    assert cmds.index("\n  setup-key") < cmds.index("\n  check") < cmds.index("\n  sync")


def test_cli_no_cluster_block_exits_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cluster, "suggest", lambda: "# suggested")
    (tmp_path / "s.yaml").write_text("llm_root: /d\nrepo_path: /r\n")
    # a single-node models file (models.cluster.yaml needs a cluster block to validate)
    (tmp_path / "m.yaml").write_text(
        "defaults:\n  vllm: {image: i, gmem_min: 0.1, gmem_max: 0.8, safety_gib: 6, ttl: 1,"
        " ready_timeout: 1, check_endpoint: /health}\nmodels: []\n")
    base = ["--models", str(tmp_path / "m.yaml"), "--settings", str(tmp_path / "s.yaml")]
    assert cli.main(base + ["cluster", "sync"]) == 1
    assert "✗ no cluster: block in settings.local.yaml" in capsys.readouterr().err
    assert cli.main(base + ["cluster", "setup-key"]) == 1
    assert cli.main(base + ["cluster", "check"]) == 1
    assert "# suggested" in capsys.readouterr().out


def test_cli_render_error_exits_1(tmp_path, capsys):
    assert cli.main(["--models", str(tmp_path / "x.yaml"), "--settings",
                     str(tmp_path / "y.yaml"), "cluster", "check"]) == 1
    assert "✗" in capsys.readouterr().err


def _store_rules(head_layers, worker_layers):
    """Head (containerd store) and worker (overlay2) report different .Id values."""
    def inspect(argv):
        remote = argv[0] == "ssh"
        if "{{.Id}}" in argv:
            return _cp(0, "sha256:606f1613" if remote else "sha256:ddca1497")
        return _cp(0, worker_layers if remote else head_layers)
    return [r if r[0] != "docker image inspect" else ("docker image inspect", inspect)
            for r in _healthy_rules()]


def test_check_image_same_layers_different_ids_passes(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    layers = '["sha256:l1","sha256:l2"]\n'
    assert cluster.check(s, models, run=Fake(_store_rules(layers, layers))) == 0
    assert "PASS  image vllm-node:latest" in capsys.readouterr().out


def test_check_image_different_layers_fails(tmp_path, capsys):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    run = Fake(_store_rules('["sha256:l1","sha256:l2"]', '["sha256:l1","sha256:l3"]'))
    assert cluster.check(s, models, run=run) == 1
    assert "FAIL  image vllm-node:latest" in capsys.readouterr().out


def test_check_and_sync_inspect_layers_not_id(tmp_path):
    s, models, _ = _settings(tmp_path)
    _make_weights(s)
    run = Fake(_healthy_rules())
    cluster.check(s, models, run=run)
    insp = [c for c in run.calls if "image inspect" in " ".join(map(str, c))]
    assert insp and all("{{json .RootFS.Layers}}" in " ".join(map(str, c)) for c in insp)
    assert not any("{{.Id}}" in " ".join(map(str, c)) for c in insp)


def test_sync_skips_image_when_layers_same_ids_differ(tmp_path):
    s, models, _ = _settings(tmp_path)
    layers = '["sha256:l1"]'
    assert cluster.sync(s, models, [], run=Fake(_store_rules(layers, layers)),
                        popen=_popen()) == 0
    assert FakePopen.instances == []
