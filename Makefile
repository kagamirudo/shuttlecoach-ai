.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
HOST ?= 127.0.0.1
PORT ?= 8501
DEVICE ?= auto
HOST_PYTHON ?= python3
DOCKER_PORT ?= 8502

export HOST PORT DEVICE CONFIG RUN VIDEO URL REVIEW OUT PLAYER_REFS SHUTTLE_REFS CHECKS
export DOCKER_PORT

.PHONY: help setup deps models run doctor test check test-ui test-calibration-ui \
 ingest probe prepare-play analyze recover-players motion-preview footwork llm-report shuttle associate-shuttle \
 validate validate-players validate-shuttle report clean check-env force docker-up docker-down docker-build docker-logs docker-status docker-test

help: ## Show commands and examples (default)
	@printf '\nShuttleCoach commands\n\n'
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nExamples:\n  make setup\n  make run PORT=8502\n  make analyze CONFIG="data/clip-config.json" DEVICE=mps\n  make shuttle RUN="results/RUN_ID" DEVICE=mps\n  make validate RUN="results/RUN_ID" PLAYER_REFS="players.csv" SHUTTLE_REFS="birdie.csv" CHECKS="checks.json"\n\n'

$(PYTHON):
	uv venv --python 3.12 .venv

setup: deps ## Install locked dependencies and download/verify both models (requires uv)
	$(MAKE) models

deps: $(PYTHON) requirements-lock.txt ## Create .venv if needed and sync locked dependencies
	uv pip sync --python "$(PYTHON)" requirements-lock.txt

models: check-env ## Download missing weights and verify pinned checksums
	"$(PYTHON)" scripts/download_model.py
	"$(PYTHON)" scripts/download_tracknet_model.py

run: check-env ## Start dashboard; HOST=127.0.0.1 PORT=8501 by default
	"$(PYTHON)" -m streamlit run app.py --server.headless true --server.address "$$HOST" --server.port "$$PORT" --browser.gatherUsageStats false

docker-up: ## Build and start Docker at localhost:8502; reuse saved Groq key and persistent media
	"$(HOST_PYTHON)" scripts/docker_compose.py up --build -d --wait

docker-down: ## Stop Docker; keep footage, models, results and cache
	"$(HOST_PYTHON)" scripts/docker_compose.py down

docker-build: ## Build the CPU Docker image without loading API credentials
	"$(HOST_PYTHON)" scripts/docker_compose.py build

docker-logs: ## Follow Docker application logs
	"$(HOST_PYTHON)" scripts/docker_compose.py logs -f --tail=100

docker-status: ## Show Docker service and health
	"$(HOST_PYTHON)" scripts/docker_compose.py ps

docker-test: ## Run unit/integration tests inside the built Docker image
	SHUTTLECOACH_NO_SECRET=1 "$(HOST_PYTHON)" scripts/docker_compose.py run --rm --no-deps app python -m pytest -q -o cache_dir=/tmp/shuttlecoach-pytest

doctor: check-env ## Check dependency consistency, Python/device details and model presence
	uv pip check --python "$(PYTHON)"
	"$(PYTHON)" -c 'import sys, platform, torch; from pathlib import Path; print("Python:", sys.version.split()[0], "Architecture:", platform.machine()); print("MPS available:", torch.backends.mps.is_available()); [print(str(p) + (": present" if p.is_file() else ": missing; run make models")) for p in map(Path, ["models/yolo11m-pose.pt", "models/TrackNet_best.pt"])]'

test: check-env ## Run the unit/integration test suite
	"$(PYTHON)" -m pytest -q

check: doctor ## Run environment checks, then tests
	$(MAKE) test

test-ui: check-env require-VIDEO require-RUN ## Check running dashboard playback/exports in installed Chrome
	SHUTTLECOACH_URL="http://$$HOST:$$PORT" "$(PYTHON)" scripts/browser_check.py "$$VIDEO" "$$(basename "$$RUN")"

test-calibration-ui: check-env ## Check cropped-court calibration in installed Chrome (dashboard must be running)
	SHUTTLECOACH_URL="http://$$HOST:$$PORT" PYTHONPATH=. "$(PYTHON)" scripts/calibration_browser_check.py

ingest: check-env require-URL ## Download one authorized video: URL="https://..."
	"$(PYTHON)" -m shuttlecoach.cli ingest "$$URL"

probe: check-env require-VIDEO ## Inspect exact source timing: VIDEO="data/clip.mp4"
	"$(PYTHON)" -m shuttlecoach.cli probe "$$VIDEO"

prepare-play: check-env require-REVIEW require-OUT ## Assemble reviewed play: REVIEW="review.json" OUT="data/selected-play"
	"$(PYTHON)" scripts/prepare_active_play.py "$$REVIEW" --out "$$OUT"

analyze: check-env require-CONFIG ## Track players from a reviewed CONFIG; DEVICE=auto|cpu|mps
	"$(PYTHON)" -m shuttlecoach.cli analyze "$$CONFIG" --device "$$DEVICE"

recover-players: check-env require-RUN ## Recover uncertain actual poses in RUN; prints a new run path
	"$(PYTHON)" -m shuttlecoach.cli recover-players "$$RUN" --device "$$DEVICE"

footwork: check-env require-RUN ## Analyze cached pose candidates; manual review required for technique timing
	"$(PYTHON)" -m shuttlecoach.cli footwork "$$RUN"

llm-report: check-env require-RUN ## Optional Groq evidence-linked AI analysis; saved key/env, free-plan quotas apply
	"$(PYTHON)" -m shuttlecoach.cli llm-report "$$RUN"

motion-preview: check-env require-RUN ## Render separate short-gap estimates; never change tracking validation
	"$(PYTHON)" -m shuttlecoach.cli motion-preview "$$RUN"

shuttle: check-env require-RUN ## Create a new birdie quality/candidate review, preserving RUN
	"$(PYTHON)" -m shuttlecoach.cli shuttle "$$RUN" --new-court-review --quality --candidates --device "$$DEVICE"

associate-shuttle: check-env require-RUN require-REVIEW ## Associate birdie candidates using reviewed activity intervals
	"$(PYTHON)" -m shuttlecoach.cli associate-shuttle "$$RUN" --review "$$REVIEW"

validate: require-RUN require-PLAYER_REFS require-SHUTTLE_REFS require-CHECKS ## Evaluate players then birdie; requires RUN, both reference CSVs and CHECKS
	$(MAKE) validate-players
	$(MAKE) validate-shuttle

validate-players: check-env require-RUN require-PLAYER_REFS require-CHECKS ## Evaluate player accuracy and unused court-line checks
	"$(PYTHON)" -m shuttlecoach.cli validate "$$RUN" --references "$$PLAYER_REFS" --checks "$$CHECKS"

validate-shuttle: check-env require-RUN require-SHUTTLE_REFS ## Evaluate birdie accuracy against the frozen reference sample
	"$(PYTHON)" -m shuttlecoach.cli validate-shuttle "$$RUN" --references "$$SHUTTLE_REFS"

report: check-env require-RUN ## Regenerate evidence report; coaching still requires both gates
	"$(PYTHON)" -m shuttlecoach.cli report "$$RUN"

clean: ## Remove Python/test caches only; preserve footage, weights, results and .cache provenance
	rm -rf .pytest_cache __pycache__ shuttlecoach/__pycache__ scripts/__pycache__ tests/__pycache__

check-env:
	@test -x "$(PYTHON)" || { printf '%s\n' 'Missing project Python. Run make setup first.'; exit 2; }

require-%: force
	@test -n "$$(printenv '$*')" || { printf '%s\n' 'Missing $*. See make help for usage.'; exit 2; }

force:
