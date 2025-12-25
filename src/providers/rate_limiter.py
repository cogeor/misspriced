import time
import asyncio

class YahooRateLimiter:
    """
    Yahoo Finance unofficial rate limits:
    - ~2000 requests/hour recommended
    - Implement exponential backoff on 429
    """
    
    def __init__(self, min_interval: float = 0.25):
        self.min_interval = min_interval
        self.last_request = 0.0
    
    def wait_if_needed(self):
        """Sync wait used by sync calls or wrappers."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
