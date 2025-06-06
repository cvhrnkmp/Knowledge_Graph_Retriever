class BaseRetriever():
    
    def __init__(self):
        pass
    
    def load_and_init(self, ragflow):
        # Call the load_init method if it is defined in the child class
        if hasattr(self, 'load_init') and callable(getattr(self, 'load_init')):
            self.load_init()
        else:
            print("No load_init method defined in this class.")
    
    def get_relevant_documents(self):
        pass
    
    