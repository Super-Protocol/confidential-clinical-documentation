# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026. Upstream's targets drove five Compose
# services under a `riva-nim` profile and built three images; there is one image
# and one service now. `scripts/validate-env.sh` checked a Riva URI and
# `scripts/bootstrap.sh` / `scripts/restart-production.sh` drove
# `infra/compose.dev.yml`; all four are gone with the things they checked.

COMPOSE := infra/compose.prod.yml
IMAGE   ?= scribe-server:dev

.PHONY: help build up down logs restart test measure seed-templates

help:
	@echo "build          build the one image (API + built UI + model weights)"
	@echo "up             run it on loopback:8000 (set LLM_MODEL, OPENAI_BASE_URL)"
	@echo "down           stop it"
	@echo "logs           follow its logs"
	@echo "test           the unit tests: no model, no audio, no GPU"
	@echo "measure        the end-to-end measurement; needs models and fixtures"
	@echo "seed-templates load the note templates into a running deployment"

build:
	docker build -f infra/docker/scribe.Dockerfile -t $(IMAGE) ..

up:
	docker compose -f $(COMPOSE) up -d

down:
	docker compose -f $(COMPOSE) down

logs:
	docker compose -f $(COMPOSE) logs -f

restart:
	docker compose -f $(COMPOSE) restart

test:
	cd apps/api && uv sync --locked --extra dev && .venv/bin/python -m pytest ../../../tests -q

measure:
	cd apps/api && .venv/bin/python ../../../tests/measure.py

seed-templates:
	cd apps/api && .venv/bin/python ../../scripts/seed_templates.py
