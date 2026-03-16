""" 
Rule Cache 
 
Provides an in-memory LRU cache for frequently matched rules. 
""" 
 
from collections import OrderedDict 
from typing import Optional 
from settings import RULE_CACHE_SIZE
 
class RuleCache: 
    """ 
    LRU Cache for rule retrieval results. 
    """ 
 
    def __init__(self, max_size: int = RULE_CACHE_SIZE): 
 
        self.cache = OrderedDict() 
        self.max_size = max_size 
 
    # ------------------------------------------------------- 
    # Create Cache Key 
    # ------------------------------------------------------- 
 
    def _make_key(self, agent_id: str, intent: str) -> str: 
        """ 
        Create unique cache key. 
        """ 
 
        return f"{agent_id}:{intent}" 
 
    # ------------------------------------------------------- 
    # Retrieve Cached Rule 
    # ------------------------------------------------------- 
 
    def get(self, agent_id: str, intent: str): 
 
        key = self._make_key(agent_id, intent) 
 
        if key in self.cache: 
 
            # Move key to end (LRU behavior) 
            self.cache.move_to_end(key) 
 
            return self.cache[key] 
 
        return None 
 
    # ------------------------------------------------------- 
    # Store Rule in Cache 
    # ------------------------------------------------------- 
 
    def put(self, agent_id: str, intent: str, rule_data): 
 
        key = self._make_key(agent_id, intent) 
 
        self.cache[key] = rule_data 
 
        # Move to end to mark as recently used 
        self.cache.move_to_end(key) 
 
        # Evict oldest if cache exceeds size 
        if len(self.cache) > self.max_size: 
            self.cache.popitem(last=False) 
 
    # ------------------------------------------------------- 
    # Clear Cache 
    # ------------------------------------------------------- 
 
    def clear(self): 
        """ 
        Clear all cached entries. 
        """ 
 
        self.cache.clear() 
 
    # ------------------------------------------------------- 
    # Cache Size 
    # ------------------------------------------------------- 
 
    def size(self) -> int: 
        """ 
        Return current cache size. 
        """ 
        return len(self.cache) 