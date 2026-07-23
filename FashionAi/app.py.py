import uvicorn
from src.api import app
from src.utils import load_config, setup_logging

logger = setup_logging("FashionAI.Main")

if __name__ == '__main__':
    config = load_config()
    host = config['api']['host']
    port = config['api']['port']
    logger.info(f"Starting FashionAI server on {host}:{port}...")
    uvicorn.run("app:app", host=host, port=port, reload=config['api']['reload'])