"""Run the local Who Speak gateway."""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("app.assistant_gateway.main:app", host="127.0.0.1", port=8020, reload=False)


if __name__ == "__main__":
    main()
