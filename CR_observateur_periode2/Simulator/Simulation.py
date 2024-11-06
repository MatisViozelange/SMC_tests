from MVC import MVCModel, MVCView, MVCController
from controllers import ASTWC, NN_based_STWC

if __name__ == "__main__":
    # Initialize model with simulation parameters
    model = MVCModel(time=20, Te=0.0005)
    
    # Set up controlers
    controler = ASTWC(model.time, model.Te, reference=None)
    NN_inner_controler = ASTWC(model.time, model.Te, reference=None)
    NN_controler = NN_based_STWC(NN_inner_controler, model.time, model.Te)
    NN_BO_Observator = NN_based_STWC(controler, model.time, model.Te)

    # Add controlers to the model
    model.add_controler('ASTWC', controler)
    model.add_controler('NN_based_ASTWC', NN_controler)
    model.add_controler('NN_BO_Observator', NN_BO_Observator)

    # Create view and controller
    view = MVCView(model)
    controller = MVCController(model, view)

    # Run the application
    controller.run()
