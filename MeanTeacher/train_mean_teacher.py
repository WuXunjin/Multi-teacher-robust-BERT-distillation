import tensorflow as tf 
from Bert.bert_model import bert_model
from MeanTeacher.losses import total_loss,ema

def data_slices(x_train,y_train,x_unlabel,batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices( (x_train[0],x_train[1], y_train) )
    train_dataset = train_dataset.shuffle( buffer_size=1024 ).batch(batch_size)

    unlabel_dataset = tf.data.Dataset.from_tensor_slices( (x_unlabel[0],x_unlabel[1]) )
    unlabel_dataset = unlabel_dataset.shuffle( buffer_size=1024 ).batch(batch_size)
    return train_dataset, unlabel_dataset

def train_mean_teacher(x_train, y_train, x_unlabel,epochs,batch_size,lr,max_len):
    train_dataset,unlabel_dataset = data_slices(x_train, y_train,x_unlabel,batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    student = bert_model('distilbert-base-uncased',max_len,lr)
    teacher_adv = bert_model('distilbert-base-uncased',max_len,lr)
    teacher_nat = bert_model('bert-base-uncased',max_len,lr) 
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
    progbar = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=['Accuracy', 'Overall_Loss'])
    step_counter = 0
    for epoch in range(epochs):
        tf.print(f'\nepoch {epoch + 1}')
        for step, ((input_ids,attention_ids, y_batch_train),(input_ids_un,attention_ids_un)) in enumerate(zip(train_dataset,unlabel_dataset)):
            with tf.GradientTape() as tape:
                loss = total_loss([input_ids,attention_ids], 
                                                y_batch_train,
                                                [input_ids_un,attention_ids_un],
                                                student, teacher_adv,teacher_nat)

            grads = tape.gradient(loss, student.trainable_weights)
            optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, student.trainable_weights) if grad is not None)
            step_counter += 1
            teacher_adv = ema(student, teacher_adv)
            logits = student([input_ids,attention_ids])
            train_acc = train_metrics(tf.argmax(y_batch_train, 1), tf.argmax(logits, 1))
            progbar.update(step, values=[('Accuracy', train_acc), ('Overall_Loss', loss)])

    return student