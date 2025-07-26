class NaiveBatcher:
    def __init__(self, window_size):
        self.window_size = window_size  # in time steps
        self.requests = []
        self.current_finished = []
        self.current_batch = []
        self.last_batch_time = -window_size  # so first batch starts at time 0

    def add_request(self, request):
        self.requests.append(request)

    def step(self, current_time):
        self.current_finished = []

        # Start a new batch every window_size steps
        if (current_time - self.last_batch_time) >= self.window_size and self.requests:
            self.current_batch = self.requests
            self.requests = []
            self.last_batch_time = current_time

        # Process the current batch
        if self.current_batch:
            for req in self.current_batch:
                req.tokens_left -= 1

            # Only finish the batch when ALL requests are done
            if all(req.tokens_left <= 0 for req in self.current_batch):
                self.current_finished = self.current_batch
                self.current_batch = []

    def collect_finished(self, current_time):
        for req in self.current_finished:
            req.finish_time = current_time
        return self.current_finished