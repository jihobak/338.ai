# Dockerfile
FROM python:3.11-slim

# 기본 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Poetry 설정 - 가상환경 생성하지 않음
# RUN poetry config virtualenvs.create false

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 종속성 파일 복사
COPY pyproject.toml poetry.lock* ./

# 소스 코드 복사
COPY src/ ./src/

# 환경변수 파일 복사
COPY .env ./.env

# 종속성 설치
RUN poetry install --no-interaction --no-ansi --all-extras


# 환경변수 설정
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE ${PORT}

# 애플리케이션 실행
# CMD ["poetry", "run", "uvicorn", "bot338.api.app:app", "--host=$HOST", "--port=$PORT"]
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

CMD ["./start.sh"]