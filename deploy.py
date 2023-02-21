import os

import baseten
import truss


# Initializing a Truss. Uncomment the following line to create a new Truss
# truss.create("./difussion_2-1/")

# Got to the `./difussion_2-1/model.py` to edit the
# Truss: have to fill in the `load` and `predict` functions.

# Load the Truss to memory
sd = truss.load("./difussion_2-1/")

# Local testing
# sd.docker_predict({"prompt": "a photo of an astronaut riding a horse on mars"})


# Deploy to baseten
baseten.configure("https://app.baseten.co")
baseten.login(os.environ["BASETEN_API_KEY"])
# One-line deploy:
baseten.deploy(sd, model_name="Text to Image", publish=True)


# Another Model
sd_img = truss.load("inpaint_2/")
# sd_img.predict({"prompt": "add a jacket", "image": res["images"][0])
baseten.deploy(sd_img, model_name="Image to Image", publish=True)
