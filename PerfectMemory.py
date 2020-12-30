import imageio
import glob

class PerfectMemory:
    route_views = []

    def __init__(self, path):
        for view_path in glob.glob(path+"/*.png"):
            view = imageio.imread(view_path)
            self.route_views.append(view)

    def get_views(self):
        return self.route_views


if __name__ == "__main__":
    pm = PerfectMemory("ant_world_image_databases/routes/ant1_route1")
    print(len(pm.get_views()))