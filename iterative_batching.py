class IterativeBatcher:
    def __init__(self):
        self.requests = []
        self.current_finished = []
    def add_request(self, request):
        self.requests.append(request)
    
    def step(self):
        self.current_finished = []
        for i in range(len(self.requests) - 1, -1, -1):
            self.requests[i].tokens_left -= 1
            if self.requests[i].tokens_left <= 0:
                self.current_finished.append(self.requests.pop(i))
                
    def collect_finished(self, current_time):
        for req in self.current_finished:
            req.finish_time = current_time
        return self.current_finished