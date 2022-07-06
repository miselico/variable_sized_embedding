import imp


import torch


class VariableSizedEmbedding(torch.Module):
    """TODO: complete documentation
        This is similar to the torch embedding moduls. However, internally only the number of dimensions for each embedding varies depending on how it was specified.
        To get the same dimension out when performing a lookup, an MLP is used to scale up the embeddings.
    """

    def __init__(self, embedding_sizes: torch.Tensor, embedding_dimension: int) -> None:
        """Create by specifiying the size for each of the embeddings, and the final (output) embedding size.

        So, if you need 5 embeddings with sizes 5, 10, 5, 10, 20 (in that order), and a final embedding size of 50, you create is using:

        VariableSizedEmbedding(torch.LongTensor([5, 10, 5, 10, 2]), 50)

        """
        super().__init__()

        self.embedding_dimension = embedding_dimension

        # first we sort the embedding sizes and REMEMBER HOW THE SORTING WENT! Check the documentation on indices.
        values, self.indices = torch.sort(embedding_sizes)

        # We need to remember in which location in our sorted list each item now is, to do this, we keep the indices of the indices
        self.inverse_indices = torch.argsort(self.indices)

        # now we need to know how many of each we have
        unique_sizes, counts = torch.unique_consecutive(
            values, return_inverse=False, return_counts=True, dim=None)

        # TODO: for each of these unique sizes, make a normal Embedding, contianing count embeddings, store them in a list
        # make sure to store them in self in a torch.nn.ParameterList

        # TODO: create the requered number of MLPs. Also store these in a torch.nn.ParameterList

        # now we rememebr the cumulative counts. They are or offsets needed for lookup in the list later
        cummulative_counts = torch.cumsum(counts)
        self.start_offsets = torch.cat(
            (torch.Tensor([0]), cummulative_counts[:-1]))
        self.end_offsets = cummulative_counts

    def reset_parameters(self) -> None:
        # TODO: reset all MLPs
        pass

    def forward(self, input: torch.Tensor):
        """Lookup the embeddings"""

        # 1 get the reversed indices

        inversed_input_indices = self.inverse_indices(input)

        # iterate of the different sizes and apply the right MLP

        # in principle we do not need to zero the tensor, because we overwrite it below but this seems useful for debugging.
        # TODO check the dimensions of this operation
        outputs = (torch.zeros_like(input)).repeat(self.embedding_dimension)

        for MLP_index, (start, end) in enumerate(zip(self.start_offsets, self.end_offsets)):
            mask = torch.logical_and(
                inversed_input_indices >= start,
                inversed_input_indices < end
            )
            indices_relevant_for_this_mlp = inversed_input_indices[mask]
            # TODO: lookup these indices in the relevant torch.Embedding, note here that we have to subtact 'start' from the indices!
            # TODO: scale the embeddings using the correct MLP
            # output_of_MLP = ...
            # TODO: insert the outputs of the MLP to the outputs
            outputs[mask] = output_of_MLP

        return outputs
