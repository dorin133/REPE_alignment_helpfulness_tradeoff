import torch
from repe.rep_control_reading_vec import WrappedReadingVecModel
from transformers import pipeline

class WrapModel:
    def __init__(self, model, tokenizer, reading_vec_data, reading_vec_labels):
        self.model = model
        self.tokenizer = tokenizer
        # self.coeff = coeff
        self.reading_vec_data = reading_vec_data
        self.reading_vec_labels = reading_vec_labels
        pca_vectors, pca_signs, layer_id = self.prepare_wrapped_model()
        self.pca_vectors = pca_vectors
        self.pca_signs = pca_signs
        self.layer_id = layer_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_wrapped_model(self):
        model_size = sum(p.numel() for p in self.model.parameters()) // 1000000000 # in billions!
        rep_token = -1
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        direction_finder_kwargs={"n_components": 1}

        rep_reader = rep_reading_pipeline.get_directions(
                                                    self.reading_vec_data, 
                                                    rep_token=rep_token,
                                                    hidden_layers=hidden_layers,
                                                    n_difference=n_difference, 
                                                    train_labels=self.reading_vec_labels, 
                                                    direction_method=direction_method, 
                                                    direction_finder_kwargs=direction_finder_kwargs
                                                )

        pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
        pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]

        layer_ids_injections = list(range(-25, -33, -1)) # 13B
        if model_size in [7, 8]: # 7B or 8B
            layer_ids_injections = list(range(-18, -23, -1))
            
        return pca_vectors, pca_signs, layer_ids_injections

    def wrap_model(self, coeff):
        #prepare RepE model
        block_name = "decoder_block"
        wrapped_model = WrappedReadingVecModel(self.model, self.tokenizer)
        wrapped_model.unwrap()
        wrapped_model.wrap_block(self.layer_id, block_name=block_name)

        activations = {}
        for layer in self.layer_id:
            v = torch.tensor(self.pca_vectors[layer]*self.pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(self.device).half()
        wrapped_model.reset()
        wrapped_model.set_controller(self.layer_id, activations, 'decoder_block')
        return wrapped_model