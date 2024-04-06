
from ptflops import get_model_complexity_info


def model_structure(model, img_size):
    blank = ' '
    print('-' * 142)
    print('|' + ' ' * 17 + 'weight name' + ' ' * 40 + '|' \
          + ' ' * 21 + 'weight shape' + ' ' * 21 + '|' \
          + ' ' * 5 + 'number' + ' ' * 5 + '|')
    print('-' * 142)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4
    macs, params = get_model_complexity_info(model, img_size, as_strings=False, print_per_layer_stat=False,
                                             verbose=False)
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 67:
            key = key + (57 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 45:
            shape = shape + (42 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 142)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:.2f} M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('The Gflops of {}: {:.2f} G'.format(model._get_name(), (2 * int(macs) * 1e-9)))
    print('-' * 142)

    return num_para * 1e-6, macs * 1e-9
