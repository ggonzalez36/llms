#!/usr/bin/env python
"""
Chapter 2 (Hands-On LLMs) executable entrypoint.

Architecture:
  - mvc/config.py      -> CLI parsing + typed config
  - mvc/model.py       -> data/model logic (tokenization, embeddings, generation)
  - mvc/presenter.py   -> console output rendering
  - mvc/controller.py  -> section orchestration
"""

from __future__ import annotations

import logging

# Silence a harmless PyTorch Windows warning about unsupported stdio redirects.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from mvc.config import load_config
from mvc.controller import Chapter2Controller


def main() -> None:
    config = load_config()
    controller = Chapter2Controller(config)
    controller.run()


if __name__ == "__main__":
    main()
