# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder

from mysite.websocket import ConsumerTemplate

# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0
        self.terminated[self.room_name] = False

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        # No inputs needed in this experiment
        return

    # This function performs a step in the experiment
    async def process_step(self):
        # We are not going to do anything in this example
        return

    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
        # Check if the game is over
        if self.step[self.room_name] > 100:
            self.terminated[self.room_name] = True
            message = 'done'
        else:
            message = 'not done'
        # Send message to room group
        # The returned value should be a dictionnary with the 
        #   type, 
        #   message, 
        #   step, 
        #   and anything else you would like to send
        return {"type": "websocket.message", 
                "message": message, 
                "step": self.step[self.room_name]}
    
    # This function takes care of anything else we need to do at the end of the request
    async def process_extras(self):
        if self.terminated[self.room_name]:
            # Delete the variables from memory
            del self.step[self.room_name]
        else:
            self.step[self.room_name] += 1
