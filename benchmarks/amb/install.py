"""Install the YantrikDB provider into an Agent Memory Benchmark checkout.

AMB resolves memory providers from a registry in
`src/memory_bench/memory/__init__.py`, so a third-party provider has to be
copied in and registered. This script does both, idempotently, against a
checkout you already have:

    git clone https://github.com/vectorize-io/agent-memory-benchmark
    cd agent-memory-benchmark && uv sync
    python /path/to/yantrikdb/benchmarks/amb/install.py .

Then:

    uv run amb run --dataset beam --split 100k --memory yantrikdb

WHY THE REGISTRY IS REWRITTEN RATHER THAN APPENDED TO
-----------------------------------------------------
Upstream's registry imports every provider module eagerly, so using ANY
provider requires EVERY provider's SDK to be installed. That is not merely
inconvenient: `hindsight-all` depends on `uvloop`, which has no Windows
build, so the harness cannot be installed on Windows at all. This rewrites
the registry to resolve `"module:Class"` lazily, which keeps every existing
provider working while making each one's dependencies its own problem.

Nothing here redistributes AMB code or data — the script edits a checkout
you obtained yourself, and the benchmark's own datasets and result files are
never copied.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

REGISTRY = '''import importlib

from .base import MemoryProvider

# Lazy registry: name -> "module:Class". Provider modules import their SDKs
# at module level (qdrant_client, mem0, hindsight, ...), so importing every
# provider here forces EVERY SDK to be installed to use ANY provider — and
# some do not install everywhere (hindsight-all -> uvloop has no Windows
# build). Resolving the class only inside get_memory_provider keeps
# `amb providers` and `--memory <x>` working with just <x>'s dependencies.
REGISTRY: dict[str, str] = {
    "vanilla": "none:NoMemoryProvider",
    "hindsight-coding": "hscoding:HsCodingProvider",
    "bm25": "bm25:BM25MemoryProvider",
    "cognee": "cognee:CogneeMemoryProvider",
    "hindsight": "hindsight:HindsightMemoryProvider",
    "hindsight-cloud": "hindsight:HindsightCloudMemoryProvider",
    "hindsight-http": "hindsight:HindsightHTTPMemoryProvider",
    "mastra": "mastra:MastraMemoryProvider",
    "mastra-om": "mastra_om:MastraOMMemoryProvider",
    "mem0": "mem0:Mem0MemoryProvider",
    "mem0-cloud": "mem0_cloud:Mem0CloudMemoryProvider",
    "ogham": "ogham:OghamMemoryProvider",
    "qdrant": "hybrid_search:HybridSearchMemoryProvider",
    "supermemory": "supermemory:SupermemoryMemoryProvider",
    "yantrikdb": "yantrikdb:YantrikDBMemoryProvider",
    "yantrikdb-floor": "yantrikdb:YantrikDBFloorMemoryProvider",
    "yantrikdb-temporal": "yantrikdb:YantrikDBTemporalMemoryProvider",
    "yantrikdb-cognitive": "yantrikdb:YantrikDBCognitiveMemoryProvider",
    "yantrikdb-rerank": "yantrikdb:YantrikDBRerankMemoryProvider",
}
# legacy aliases (docs/scripts used these); canonical names above
REGISTRY["none"] = REGISTRY["vanilla"]
REGISTRY["hscoding"] = REGISTRY["hindsight-coding"]


def resolve_provider_class(name: str) -> type[MemoryProvider]:
    if name not in REGISTRY:
        raise ValueError(f"Unknown memory provider: '{name}'. Available: {list(REGISTRY)}")
    module_name, _, class_name = REGISTRY[name].partition(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, class_name)


def get_memory_provider(name: str) -> MemoryProvider:
    return resolve_provider_class(name)()
'''

LLM_REGISTRATION = (
    "from .ollama import OllamaLLM\n",
    '    "ollama": OllamaLLM,\n',
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("amb_root", type=Path, help="path to an agent-memory-benchmark checkout")
    a = ap.parse_args()

    root = a.amb_root.resolve()
    mem = root / "src" / "memory_bench" / "memory"
    llm = root / "src" / "memory_bench" / "llm"
    if not mem.is_dir() or not llm.is_dir():
        print(f"error: {root} does not look like an AMB checkout "
              f"(expected src/memory_bench/{{memory,llm}})", file=sys.stderr)
        return 1

    shutil.copy2(HERE / "yantrikdb.py", mem / "yantrikdb.py")
    shutil.copy2(HERE / "ollama.py", llm / "ollama.py")
    shutil.copy2(HERE / "frozen_context_eval.py", root / "frozen_context_eval.py")
    shutil.copy2(HERE / "frozen_stats.py", root / "frozen_stats.py")
    print(f"copied provider  -> {mem / 'yantrikdb.py'}")
    print(f"copied llm       -> {llm / 'ollama.py'}")
    print(f"copied evaluator -> {root / 'frozen_context_eval.py'}")
    print(f"copied stats     -> {root / 'frozen_stats.py'}")

    (mem / "__init__.py").write_text(REGISTRY, encoding="utf-8")
    print(f"rewrote registry -> {mem / '__init__.py'} (lazy; all upstream providers preserved)")

    # Register the ollama LLM, idempotently.
    llm_init = llm / "__init__.py"
    src = llm_init.read_text(encoding="utf-8")
    if "OllamaLLM" not in src:
        imp, entry = LLM_REGISTRATION
        src = src.replace("from .openai import OpenAILLM\n",
                          "from .openai import OpenAILLM\n" + imp, 1)
        src = src.replace('    "openai": OpenAILLM,\n',
                          entry + '    "openai": OpenAILLM,\n', 1)
        llm_init.write_text(src, encoding="utf-8")
        print(f"registered ollama LLM -> {llm_init}")
    else:
        print("ollama LLM already registered")

    print("\ndone. verify with:\n  uv run amb providers | grep yantrikdb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
