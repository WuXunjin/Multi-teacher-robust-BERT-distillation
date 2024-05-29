import tensorflow as tf 
import numpy as np 

def total_loss(x_train,y_train,x_unlabel,student, teacher_adv, teacher_nat):
    temperature=3.0
    logits_s = student(x_train)
    logits_t_soft = tf.nn.softmax(teacher_nat(x_train) / temperature)
    classification_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(logits_s,y_train))
    distillation_loss = tf.keras.losses.categorical_crossentropy(logits_s, logits_t_soft)
    logits_s_adv = student(x_unlabel)
    logits_t_adv = teacher_adv(x_unlabel)
    consistency_loss = tf.reduce_sum(tf.losses.mean_squared_error(logits_s_adv,logits_t_adv))
    return 0.4*classification_loss+0.3*distillation_loss+0.3*consistency_loss
    #return 0.5*classification_loss+0.5*distillation_loss
    #return 0.5*classification_loss+0.5*consistency_loss
    #return classification_loss

def ema(student_model, teacher_model) :
    # taking weights
    student_weights = student_model.get_weights()
    teacher_weights = teacher_model.get_weights()
    # length must be equal otherwise it will not work
    assert len(student_weights ) == len(teacher_weights ), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format (
        len(student_weights ), len (teacher_weights ) )
    new_layers = []
    for i, layers in enumerate ( student_weights ) :
        new_layer = 0.999 * (teacher_weights[i]) + 0.001 * layers
        new_layers.append(new_layer)
    teacher_model.set_weights(new_layers)
    return teacher_model