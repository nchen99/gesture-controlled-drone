"""
A general window-based filter to prune volatile results in a period of frames using a Queue implementation.
Queues the inference result from ModelProcessor then return the most frequent output label in the queue, empty Queue 
and repeat. 

:parameters:
    fsp         - fps model is able to output inference result 
    window      - how long the filter should last for in seconds

How to use:
import into RunLive and instantiate an instance of DecisionFilter based on Model specification (fps) and 
user specification (window)

from .DecisionFilter import DecisionFilter
filter = DecisionFilter(fps, window)
result_img, action = self.model_processor.predict(frame_org)
distilled_result = filter.sample(action)
"""

from queue import Queue
import numpy as np


class DecisionFilter:
    def __init__(self, fps=4, window=3, **kwargs):
        self.inference_tracker = dict()
        self.mode_inference = None
        self.fps = fps if "fps" not in kwargs else kwargs["fps"]
        self.window = window if "window" not in kwargs else kwargs["window"]
        self.max_qsize = int(self.fps * self.window)
        self.q = Queue(maxsize=self.max_qsize) 


    def sample(self, result):
        if not self.q.full():
            self._enqueue(result)
            return "MODE_INFERENCE_SAMPLING"
        else:
            # get the mode result, then release
            self.mode_inference = self._release()
            print(f"Mode inference result: {self.mode_inference}")
            return self.mode_inference
        
    def _enqueue(self, result):
        """
        Put inference result to the Queue when there is room and create dynamic hashtable
        of inference result. (Key=Inference Output, Value=Count) each enqueue will update the hashtable
        """
        self.q.put(result)

        if result in self.inference_tracker:
            self.inference_tracker[result] += 1
        else:
            print(f"Detected new gesture: {result}")
            self.inference_tracker[result] = 1

    def _release(self):
        """Release elements in the queue and get the key from inference_tracker with the highest value"""
        # distilled_result = max(self.inference_tracker, key=self.inference_tracker.get)
        distilled_result = None
        if "Presence" in self.inference_tracker and (self.inference_tracker["Presence"] / self.max_qsize) >= 0.6:
            distilled_result = "Presence"
        
        print("Release: ", self.inference_tracker)
        print("Max queue size reached, release elements in queue for next batch, reset hash table...")
        with self.q.mutex:
            self.q.queue.clear()
        self.inference_tracker = dict()
        return distilled_result
    
    def __repr__(self):
        return f"DecisionFilter(fps={self.fps}, window={self.window})"


    