import numpy as np
import tensorflow as tf
from lime import lime_tabular, lime_image


def LIME(img, dataset, model, model_type):
    if model_type == "1d":
        explainer = lime_tabular.LimeTabularExplainer(dataset['x'],
                                                      feature_names=list(range(40)),
                                                      class_names=list(range(10)))
        explanation = explainer.explain_instance(img, model.predict, num_features=40, top_labels=1)

        lime_mask = list(explanation.local_exp.values())[0]
        lime_mask.sort(key=lambda i: i[0])
        lime_mask = np.transpose(np.array([m[1] for m in lime_mask]))
        lime_mask -= np.min(lime_mask)
        lime_mask /= np.max(lime_mask)
    else:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img, model.predict, top_labels=1, hide_color=0, num_samples=1000)
        lime_img, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=5)
    return lime_mask


def compute_gradients(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def interpolate_images(baseline,
                       image,
                       alphas,
                       model_type):
    if model_type == '1d':
        alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    else:
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def one_batch(baseline, image, model, model_type, alpha_batch, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch,
                                                       model_type=model_type)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx,
                                       model=model)
    return gradient_batch


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(image,
                         model,
                         model_type,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
    baseline = tf.zeros(shape=image.shape)

    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Collect gradients.
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(baseline, image, model, model_type, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = tf.stack(gradient_batch)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    integrated_gradients -= np.min(integrated_gradients)
    integrated_gradients /= np.max(integrated_gradients)
    return integrated_gradients.numpy()
