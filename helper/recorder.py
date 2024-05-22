from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from pymoo.visualization.scatter import Scatter


def record(res, filename="Res.mp4"):
    # use the video writer as a resource
    with Recorder(Video(filename)) as rec:

        # for each algorithm object in the history
        for entry in res.history:
            sc = Scatter(title=("Gen %s" % entry.n_gen))
            sc.add(entry.pop.get("F"))
            sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            sc.do()

            # finally record the current visualization to the video
            rec.record()