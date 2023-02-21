import os

import baseten
import truss

baseten.configure("https://app.baseten.co")
baseten.login(os.environ["BASETEN_API_KEY"])


# Truss init. Uncomment the following line to create a new truss
# truss.create("./difussion_2-1/")

# Got to the `./difussion_2-1/model.py` so that you
# edit it to load and run predict using your model

# Deploy first model
sd = truss.load("./difussion_2-1/")

# Local Testing
# sd.docker_predict({"prompt": "a photo of an astronaut riding a horse on mars"})

# Deploy to baseten in one line
baseten.deploy(sd, model_name="Text to Image", publish=True)

# Another Model
sd_img = truss.load("inpaint_2/")
# sd_img.predict({"prompt": "add a jacket", "image": res["images"][0])
baseten.deploy(sd_img, model_name="Image to Image", publish=True)
