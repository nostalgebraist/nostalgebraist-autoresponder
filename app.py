import json
import sys
import tensorflow as tf
sys.path.append("gpt-2/")
sys.path.append("gpt-2/src/")
import tflex
import model
import encoder


def lambda_handler(event, context):
    tf.reset_default_graph()

    session = tflex.Session()

    context = tf.placeholder(tf.int32, [1, None])

    # hparams = model.hparams_1558M()
    hparams = model.default_hparams()

    logits_op = model.model(hparams=hparams, X=context)[
                "logits"
            ]

    saver = tflex.Saver()
    # ckpt = 'models/autoresponder_v10_1/model-141.hdf5'
    ckpt = tflex.latest_checkpoint('models/117M/')

    enc = encoder.get_encoder_from_path('autoresponder_v10_1/')
    saver.restore(session, ckpt)

    toks = enc.encode("<|endoftext|>")
    logits = session.run(logits_op, feed_dict={context: [toks]})

    shape = [x for x in logits.shape]

    return {
        'statusCode': 200,
        'body': json.dumps({'logit_shape': shape})
    }
