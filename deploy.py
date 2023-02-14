import baseten
import truss

# Deploy first model
sd = truss.load("./sd_v2-1_truss/")

# Local Testing
# sd.docker_predict({"prompt": "a photo of an astronaut riding a horse on mars"})

# Deploy to baseten in one line
baseten.deploy(sd, model_name="Text to Image", publish=True)

# Another Model
sd_img = truss.load("sd_v2-1_img_truss/")
# sd_img.predict({"prompt": "add a jacket", "image": res["images"][0])
baseten.deploy(sd_img, model_name="Image to Image", publish=True)
