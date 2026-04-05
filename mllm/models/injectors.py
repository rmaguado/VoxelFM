import torch
from mllm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX


class ImageTokenInjector:
    def __call__(
        self, input_ids, labels, attention_mask, image_embeds, embed_tokens, dtype
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        new_embeds, new_labels = [], []

        for b in range(batch_size):
            ids = input_ids[b][attention_mask[b]]
            lbl = labels[b][attention_mask[b]]

            if image_embeds[b] is None:
                embeds = embed_tokens(ids)
                new_embeds.append(embeds)
                new_labels.append(lbl)
                continue

            img_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            assert len(img_pos) == 1, "Only one <image> token supported"

            i = img_pos.item()
            before = embed_tokens(ids[:i])
            after = embed_tokens(ids[i + 1 :])

            img = image_embeds[b]  # (num_img_tokens, hidden)

            new_embeds.append(torch.cat([before, img, after], dim=0))
            new_labels.append(
                torch.cat(
                    [
                        lbl[:i],
                        torch.full(
                            (img.size(0),),
                            IGNORE_INDEX,
                            device=device,
                            dtype=lbl.dtype,
                        ),
                        lbl[i + 1 :],
                    ],
                    dim=0,
                )
            )

        max_len = max(x.size(0) for x in new_embeds)
        hidden = new_embeds[0].size(-1)

        padded_embeds = torch.zeros(
            batch_size, max_len, hidden, device=device, dtype=dtype
        )
        padded_labels = torch.full(
            (batch_size, max_len), IGNORE_INDEX, device=device, dtype=labels.dtype
        )
        attn_out = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for i, (e, l) in enumerate(zip(new_embeds, new_labels)):
            padded_embeds[i, : e.size(0)] = e
            padded_labels[i, : l.size(0)] = l
            attn_out[i, : e.size(0)] = True

        return padded_embeds, padded_labels, attn_out
