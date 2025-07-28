class Request:
    """
    Represents an LLM request in the simulation.
    """
    def __init__(self, request_id, arrival_time, tokens_needed):
        self.id = request_id
        self.arrival_time = arrival_time
        self.tokens_left = tokens_needed
        self.finish_time = None
        self.queue_time = 0  # Track how long request has been waiting in queue