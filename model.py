import tensorflow as tf
from tensorflow.keras import layers, Model


def build_visual_encoder(image_size=224, feat_dim=512, backbone='simple_cnn'):
    """
    Lightweight CNN visual encoder that replaces ResNet50.
    Produces a fixed feature vector of dimension feat_dim.
    """
    inp = tf.keras.Input(shape=(image_size, image_size, 3), name='image_input')
    x = inp

    # --- Stem ---
    x = layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)

    # --- Block 1 ---
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)

    # --- Block 2 ---
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)

    # --- Block 3 ---
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # --- Projection layer ---
    x = layers.Dense(feat_dim, activation='relu', name='proj_dense')(x)

    return Model(inputs=inp, outputs=x, name='visual_encoder_cnn')


def build_text_encoder(vocab_size, embed_dim=300, hidden_dim=512, max_len=64):
    """
    Encodes a caption into a fixed-size hidden vector (LSTM final hidden state).
    mask_zero=False ensures no mask propagation issues.
    """
    inp = tf.keras.Input(shape=(max_len,), dtype='int32', name='caption_input')
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=False)(inp)
    _, state_h, _ = layers.LSTM(hidden_dim, return_state=True)(x)
    return Model(inputs=inp, outputs=state_h, name='text_encoder')


def build_multimodal_model(cfg):
    seq_len = cfg['dataset']['seq_len']
    image_size = cfg['dataset']['image_size']
    max_cap_len = cfg['dataset']['max_caption_len']

    # Build encoders
    visual_enc = build_visual_encoder(image_size=image_size,
                                      feat_dim=cfg['model']['image_feat_dim'])
    text_enc = build_text_encoder(vocab_size=cfg['model']['vocab_size'],
                                  embed_dim=cfg['model']['text_embed_dim'],
                                  hidden_dim=cfg['model']['text_hidden_dim'],
                                  max_len=max_cap_len)

    # Inputs
    images_in = tf.keras.Input(shape=(seq_len, image_size, image_size, 3), name='images_seq')
    captions_in = tf.keras.Input(shape=(seq_len, max_cap_len), dtype='int32', name='captions_seq')

    # Encode each timestep
    td_visual = layers.TimeDistributed(visual_enc, name='td_visual')(images_in)
    td_text = layers.TimeDistributed(text_enc, name='td_text')(captions_in)

    # Fuse and project
    fused = layers.Concatenate(axis=-1, name='fuse_vis_text')([td_visual, td_text])
    fused = layers.TimeDistributed(
        layers.Dense(cfg['model']['multimodal_dim'], activation='relu'),
        name='td_proj'
    )(fused)

    # Temporal LSTM
    temporal_out, _, _ = layers.LSTM(
        cfg['model']['temporal_hidden_dim'],
        return_sequences=True,
        return_state=True,
        name='temporal_lstm'
    )(fused)

    # Last timestep context
    context = temporal_out[:, -1, :]

    # --- Text Decoder ---
    dec_input = tf.keras.Input(shape=(max_cap_len,), dtype='int32', name='dec_input')
    dec_emb = layers.Embedding(cfg['model']['vocab_size'],
                               cfg['model']['text_embed_dim'],
                               mask_zero=False)(dec_input)

    ctx_tile = layers.RepeatVector(max_cap_len)(context)
    dec_lstm_in = layers.Concatenate()([dec_emb, ctx_tile])
    dec_lstm_out = layers.LSTM(cfg['model']['text_decoder_hidden'], return_sequences=True)(dec_lstm_in)
    logits = layers.TimeDistributed(layers.Dense(cfg['model']['vocab_size']))(dec_lstm_out)

    text_decoder_model = tf.keras.Model(
        inputs=[images_in, captions_in, dec_input], outputs=logits, name='text_decoder'
    )

    # --- Image feature predictor ---
    img_pred = layers.Dense(cfg['model']['temporal_hidden_dim'], activation='relu')(context)
    img_pred = layers.Dense(cfg['model']['image_feat_dim'])(img_pred)

    # Final multimodal model
    full_model = tf.keras.Model(
        inputs=[images_in, captions_in, dec_input],
        outputs=[logits, img_pred],
        name='multimodal_model'
    )

    return {
        "full_model": full_model,
        "visual_enc": visual_enc,
        "text_enc": text_enc,
        "text_decoder": text_decoder_model
    }
