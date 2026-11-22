# LedgerTrap: offline, CPU-only, standard library only.
# Build:  docker build -t ledgertrap .
# Check:  docker run --rm --network none --cpus 2 --memory 4g ledgertrap
# Other:  docker run --rm --network none ledgertrap python -m ledgertrap jury
# Jury adapter (official checks of 20.11.2026): /app/judge_adapter.py, working directory /app. The mounted
# folder must be writable by uid 10001, or run with your own uid:
#         docker run --rm --network none --user "$(id -u):$(id -g)" -v "$PWD/jury:/jury" ledgertrap \
#           python -c "import judge_adapter as j; j.generate_episode(1, {}, '/jury/ep1')"
FROM python:3.11.18-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LANG=C.UTF-8

# Unprivileged user; the repository has no third-party dependencies to install.
# /app is created owned by `runner` (WORKDIR alone would create it as root, and `check` writes /app/build).
RUN useradd --create-home --uid 10001 runner && install -d -o runner -g runner /app
WORKDIR /app
COPY --chown=runner:runner . /app
USER runner

CMD ["python", "-m", "ledgertrap", "check"]
