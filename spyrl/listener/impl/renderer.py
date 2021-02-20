from spyrl.listener.step_listener import StepListener
from spyrl.util.util import override

class Renderer(StepListener):
    @override(StepListener)
    def after_step(self, event):
        event.env.render()