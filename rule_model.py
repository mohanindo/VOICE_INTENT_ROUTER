""" 
Rule Models 
 
Defines structured objects used across the SourceBytes Agent Engine. 
""" 
 
from dataclasses import dataclass, field 
from typing import List, Optional, Dict 
 
 
# ------------------------------------------------------- 
# Rule Model (Parsed from Excel) 
# ------------------------------------------------------- 
 
@dataclass 
class Rule: 
    """ 
    Represents a rule parsed from Excel configuration. 
    """ 
 
    agent_id: str 
    rule_id: str 
 
    category: str 
    subcategory: Optional[str] = None 
    sub_subcategory: Optional[str] = None 
 
    intent_examples: List[str] = field(default_factory=list) 
    bot_questions: List[str] = field(default_factory=list) 
 
    severity: Optional[str] = None 
    workflow: Optional[str] = None 
    escalation: Optional[str] = None 
    resolution: Optional[str] = None 
 
    metadata: Dict = field(default_factory=dict) 
 
 
# ------------------------------------------------------- 
# Vector Record (Stored in FAISS) 
# ------------------------------------------------------- 
 
@dataclass 
class VectorRecord: 
    """ 
    Represents an embedding stored in the vector database. 
    """ 
 
    text: str 
    embedding: List[float] 
 
    agent_id: str 
    rule_id: str 
 
    metadata: Dict 
 
 
# ------------------------------------------------------- 
# Rule Match (Vector Search Result) 
# ------------------------------------------------------- 
 
@dataclass 
class RuleMatch: 
    """ 
    Represents a rule returned from vector search. 
    """ 
 
    rule_id: str 
    score: float 
    metadata: Dict 
 
 
# ------------------------------------------------------- 
# Decision Result (Output of Decision Engine) 
# ------------------------------------------------------- 
 
@dataclass 
class DecisionResult: 
    """ 
    Represents final decision returned by the decision engine. 
    """ 
 
    rule_id: str 
 
    category: Optional[str] = None 
    subcategory: Optional[str] = None 
 
    severity: Optional[str] = None 
    workflow: Optional[str] = None 
    escalation: Optional[str] = None 
 
    bot_question: Optional[str] = None 
 
    metadata: Dict = field(default_factory=dict) 