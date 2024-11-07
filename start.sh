#!/bin/bash

poetry run uvicorn bot338.api.app:app --host="0.0.0.0" --port=8000