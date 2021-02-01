import airsim

client = airsim.VehicleClient()
client.confirmConnection()

for i in range(10):
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanner, True, False),
    ])