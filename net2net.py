import numpy as np

class Net2Net(object):
    def __init__(self, error=1e-3):
        self._error_th = error
        print ('Net2Net module initialize...')

    def deeper(self, weight, verification=True):
        """ Net2Deeper operation
          
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)

        Args:
            weight: weight matrix where the layer to be deepened

        Returns:
            Identity matrix & bias fitted to input weight
        """
        assert weight.ndim == 4 or weight.ndim == 2, 'Check weight.ndim'
        if weight.ndim == 2:
            deeper_w = np.eye(weight.shape[1])
            deeper_w += np.random.normal(loc=0, scale=1e-7, size=deeper_w.shape)
            deeper_b = np.zeros(weight.shape[1])
            if verification:
                err = np.abs(np.sum(np.dot(weight, deeper_w)-weight))
                assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        else:
            deeper_w = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3], weight.shape[3]))
            assert weight.shape[0] % 2 == 1 and weight.shape[1] % 2 == 1, 'Kernel size should be odd'
            center_h = (weight.shape[0]-1)//2
            center_w = (weight.shape[1]-1)//2
            for i in range(weight.shape[3]):
                tmp = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3]))
                tmp[center_h, center_w, i] = 1
                deeper_w[:, :, :, i] = tmp
            deeper_b = np.zeros(weight.shape[3])
            if verification:
                import scipy.signal
                inputs = np.random.rand(weight.shape[0]*4, weight.shape[1]*4, weight.shape[2])
                ori = np.zeros((weight.shape[0]*4, weight.shape[1]*4, weight.shape[3]))
                new = np.zeros((weight.shape[0]*4, weight.shape[1]*4, weight.shape[3]))
                for i in range(weight.shape[3]):
                    for j in range(inputs.shape[2]):
                        if j==0: tmp = scipy.signal.convolve2d(inputs[:,:,j], weight[:,:,j,i], mode='same')
                        else: tmp += scipy.signal.convolve2d(inputs[:,:,j], weight[:,:,j,i], mode='same')
                    ori[:,:,i] = tmp
                for i in range(deeper_w.shape[3]):
                    for j in range(ori.shape[2]):
                        if j==0: tmp = scipy.signal.convolve2d(ori[:,:,j], deeper_w[:,:,j,i], mode='same')
                        else: tmp += scipy.signal.convolve2d(ori[:,:,j], deeper_w[:,:,j,i], mode='same')
                    new[:,:,i] = tmp
                err = np.abs(np.sum(ori-new))
                assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return deeper_w, deeper_b 

    def wider(self, weight1, bias1, weight2, new_width, verification=True):
        """ Net2Wider operation
        
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        
        Args:    
            weight1: weight matrix of a target layer
            bias1: biases of a target layer, bias1.ndim = 1
            weight2: weight matrix of a next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (w1, b1, w2)
        """
        # Check dimensions
        assert bias1.squeeze().ndim==1, 'Check bias.ndim'
        assert weight1.ndim == 4 or weight1.ndim == 2, 'Check weight1.ndim'
        assert weight2.ndim == 4 or weight2.ndim == 2, 'Check weight2.ndim'
        bias1 = bias1.squeeze()
        if weight1.ndim == 2:
            assert weight1.shape[1] == weight2.shape[0], 'Check shape of weight'
            assert weight1.shape[1] == len(bias1), 'Check shape of bias'
            assert weight1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc(weight1, bias1, weight2, new_width, verification)
        else:
            assert weight1.shape[3] == weight2.shape[2], 'Check shape of weight'
            assert weight1.shape[3] == len(bias1), 'Check shape of bias'
            assert weight1.shape[3] < new_width, 'new_width should be larger than old width'
            return self._wider_conv(weight1, bias1, weight2, new_width, verification)

    def wider_lstm(self, kernel, bias, weight2_list, new_width, verification=True):
        """ Net2Wider operation
        
        All weights & biases should be 'numpy' array.
        
        Args:    
            kernel: lstm_kernel (2*input_size) * (4*output_size)
            bias1: lstm_kernel 4*output_size
            weight2: a list, weight matrix of multi next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (k, b, w2)
        """
        # Check dimensions
        assert bias.squeeze().ndim==1, 'Check bias.ndim'
        assert kernel.ndim == 2, 'Check kernel.ndim'
        for weight2 in weight2_list:
            assert weight2.ndim == 2, 'Check weight2.ndim'
        bias = bias.squeeze()
        for weight2 in weight2_list:
            assert kernel.shape[1]/4 == weight2.shape[0], 'Check shape of weight'
        assert kernel.shape[1] == len(bias), 'Check shape of bias'
        assert kernel.shape[1]/4 < new_width, 'new_width should be larger than old width'
        return self._wider_lstm(kernel, bias, weight2_list, new_width, verification)

    def wider_multi(self, weight1, bias1, weight2_list, new_width, verification=True):
        """ Net2Wider operation
        
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        
        Args:    
            weight1: weight matrix of a target layer
            bias1: biases of a target layer, bias1.ndim = 1
            weight2: a list, weight matrix of multi next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (w1, b1, w2)
        """
        # Check dimensions
        assert bias1.squeeze().ndim==1, 'Check bias.ndim'
        assert weight1.ndim == 4 or weight1.ndim == 2, 'Check weight1.ndim'
        for weight2 in weight2_list:
            assert weight2.ndim == 4 or weight2.ndim == 2, 'Check weight2.ndim'
        bias1 = bias1.squeeze()
        if weight1.ndim == 2:
            for weight2 in weight2_list:
                assert weight1.shape[1] == weight2.shape[0], 'Check shape of weight'
            assert weight1.shape[1] == len(bias1), 'Check shape of bias'
            assert weight1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc_multi(weight1, bias1, weight2_list, new_width, verification)
        else:
            for weight2 in weight2_list:
                assert weight1.shape[3] == weight2.shape[2], 'Check shape of weight'
            assert weight1.shape[3] == len(bias1), 'Check shape of bias'
            assert weight1.shape[3] < new_width, 'new_width should be larger than old width'
            return NotImplemented
            return self._wider_conv_multi(weight1, bias1, weight2_list, new_width, verification)
    
    def wider_rand(self, weight1, bias1, weight2, new_width):
        """ Net2Wider operation with random pad (baseline)
        
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        
        Args:    
            weight1: weight matrix of a target layer
            bias1: biases of a target layer, bias1.ndim = 1
            weight2: weight matrix of a next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (w1, b1, w2)
        """
        # Check dimensions
        assert bias1.squeeze().ndim==1, 'Check bias.ndim'
        assert weight1.ndim == 4 or weight1.ndim == 2, 'Check weight1.ndim'
        assert weight2.ndim == 4 or weight2.ndim == 2, 'Check weight2.ndim'
        bias1 = bias1.squeeze()
        if weight1.ndim == 2:
            assert weight1.shape[1] == weight2.shape[0], 'Check shape of weight'
            assert weight1.shape[1] == len(bias1), 'Check shape of bias'
            assert weight1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc_rand(weight1, bias1, weight2, new_width)
        else:
            assert weight1.shape[3] == weight2.shape[2], 'Check shape of weight'
            assert weight1.shape[3] == len(bias1), 'Check shape of bias'
            assert weight1.shape[3] < new_width, 'new_width should be larger than old width'
            return self._wider_conv_rand(weight1, bias1, weight2, new_width)
           
    def _wider_conv(self, teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        rand = np.random.randint(teacher_w1.shape[3], size=(new_width-teacher_w1.shape[3]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_weight = teacher_w1[:, :, :, teacher_index]
            new_weight = new_weight[:, :, :, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=3)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            new_weight = teacher_w2[:, :, teacher_index, :]*(1./factor)
            new_weight_re = new_weight[:, :, np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
            student_w2[:, :, teacher_index, :] = new_weight
        if verification:
            import scipy.signal
            inputs = np.random.rand(teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, teacher_w1.shape[2])
            ori1 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, teacher_w1.shape[3]))
            ori2 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, teacher_w2.shape[3]))
            new1 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, student_w1.shape[3]))
            new2 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, student_w2.shape[3]))
            for i in range(teacher_w1.shape[3]):
                for j in range(inputs.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(inputs[:,:,j], teacher_w1[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(inputs[:,:,j], teacher_w1[:,:,j,i], mode='same')
                ori1[:,:,i] = tmp + teacher_b1[i]
            for i in range(teacher_w2.shape[3]):
                for j in range(ori1.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(ori1[:,:,j], teacher_w2[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(ori1[:,:,j], teacher_w2[:,:,j,i], mode='same')
                ori2[:,:,i] = tmp
            for i in range(student_w1.shape[3]):
                for j in range(inputs.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(inputs[:,:,j], student_w1[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(inputs[:,:,j], student_w1[:,:,j,i], mode='same')
                new1[:,:,i] = tmp + student_b1[i]
            for i in range(student_w2.shape[3]):
                for j in range(new1.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(new1[:,:,j], student_w2[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(new1[:,:,j], student_w2[:,:,j,i], mode='same')
                new2[:,:,i] = tmp
            err = np.abs(np.sum(ori2-new2))
            assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return student_w1, student_b1, student_w2

    def _wider_conv_rand(self, teacher_w1, teacher_b1, teacher_w2, new_width):
        size = new_width-teacher_w1.shape[3]
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(size):
            shape = teacher_w1[:,:,:,0].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight = new_weight[:, :, :, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=3)
            student_b1 = np.append(student_b1, 0.1)
        # next layer update (i+1)
        for i in range(size):
            shape = teacher_w2[:,:,0,:].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight_re = new_weight[:, :, np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
        return student_w1, student_b1, student_w2
       
    def insert_fc_out(self, weight, bias, new_width, verification=True):
        assert weight.ndim == 2, "check weight.ndim"       
        old_shape = weight.shape
        new_shape = [old_shape[0], new_width]
        idx = old_shape[1]
        # assert idx < old_shape[1], "check insert idx"       

        new_w = np.zeros(new_shape)
        new_bias = np.zeros(new_shape[1])
        # for i in range(0, idx):
        #     new_w[:,i:i+1] = weight[:,i:i+1]
        #     new_bias[i] = bias[i]
        # for i in range(idx, new_shape[1]):
        #     new_w[:,i:i+1] = (np.random.rand(old_shape[0], 1) - 0.5) * 0.5
        #     new_bias[i] = (np.random.rand() - 0.5) * 0.5
        
        for i in range(0, new_shape[1]):  # 意义不同, 尝试重新从0训练
            new_w[:,i:i+1] = (np.random.rand(old_shape[0], 1) - 0.5) * 0.5
            new_bias[i] = (np.random.rand() - 0.5) * 0.5

        return new_w, new_bias

    def insert_fc_out_adam(self, weight, bias, new_width, verification=True):
        assert weight.ndim == 2, "check weight.ndim"       
        old_shape = weight.shape
        new_shape = [old_shape[0], new_width]
        idx = old_shape[1]
        # assert idx < old_shape[1], "check insert idx"       

        new_w = np.zeros(new_shape)
        new_bias = np.zeros(new_shape[1])
        # for i in range(0, idx):
        #     new_w[:,i:i+1] = weight[:,i:i+1]
        #     new_bias[i] = bias[i]
        # for i in range(idx, new_shape[1]):
        #     new_w[:,i:i+1] = (np.random.rand(old_shape[0], 1) - 0.5) * 0.5
        #     new_bias[i] = (np.random.rand() - 0.5) * 0.5
        
        for i in range(0, new_shape[1]):  # 意义不同, 尝试重新从0训练
            new_w[:,i:i+1] = (np.random.rand(old_shape[0], 1) - 0.5) * 0.5 * 1e-8
            new_bias[i] = (np.random.rand() - 0.5) * 0.5 * 1e-8

        return new_w, new_bias

    # def insert_fc_out(self, weight, bias, idx, verification=True):
    #     assert weight.ndim == 2, "check weight.ndim"       
    #     old_shape = weight.shape
    #     new_shape = [old_shape[0], old_shape[1]+1]
    #     assert idx < old_shape[1], "check insert idx"       

    #     new_w = np.zeros(new_shape)
    #     new_bias = np.zeros(new_shape[1])
    #     for i in range(0, idx):
    #         new_w[:,i:i+1] = weight[:,i:i+1]
    #         new_bias[i] = bias[i]
    #     for i in range(idx+1, new_shape[1]):
    #         new_w[:,i:i+1] = weight[:,i-1:i]
    #         new_bias[i] = bias[i-1]

    #     return new_w, new_bias

    def wider_obs(self, weight, new_width, verification=True):
        #assert weight.ndim == 4 or weight.ndim == 2, "check weight.ndim"       
        assert weight.ndim == 2, "check weight.ndim"       
        old_shape = weight.shape
        new_shape = [new_width, old_shape[1]]
        aixs_list = []
        constant_list = []
        ## 填充相同值时使用注释方法, 但是实际运行会出现参数nan的报错
        # for i in range(len(new_shape)):
        #     assert new_shape[i] >= old_shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i], old_shape[i])
        #     tmp_axis = (0, new_shape[i] - old_shape[i])
        #     aixs_list.append(tmp_axis)
        #     constant_list.append(0.0)
        # new_weight = np.pad(weight, tuple(aixs_list), 'constant', constant_values=tuple(constant_list))  # mode = constant, 表示填充相同的值, 在这里踩了个坑
        new_weight = np.zeros(new_shape)
        for i in range(0, old_shape[0]):
            new_weight[i:i+1,:] = weight[i:i+1,:]
        for i in range(old_shape[0], new_width):
            new_weight[i:i+1,:] = (np.random.rand(1, old_shape[1]) - 0.5) * 0.5
        # print("hunky debug, ", old_shape, old_shape[1], new_width, new_shape, new_weight.shape, len(constant_list))
        #new_weight += np.random.normal(loc=0, scale=1e-6, size=new_weight.shape)
        if verification:
            old_input = np.random.rand(1,weight.shape[0])
            new_input = np.hstack([old_input, np.random.rand(1,new_width-weight.shape[0])])
            old_output = np.matmul(old_input,weight)
            new_output = np.matmul(new_input,new_weight)
            assert np.mean(new_output-old_output) < 10-4, "obs wider error!"
        return new_weight

    def wider_obs2(self, old_weight, start, end, new_width, verification=True):
        assert old_weight.ndim == 2, "check weight.ndim"       

        weight = old_weight[start:end]

        old_shape = weight.shape
        new_shape = [new_width, old_shape[1]]
        aixs_list = []
        constant_list = []
        for i in range(len(new_shape)):
            assert new_shape[i] >= old_shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i], old_shape[i])
            tmp_axis = (0, new_shape[i] - old_shape[i])
            aixs_list.append(tmp_axis)
            constant_list.append(0.0)
        new_weight = np.pad(weight, tuple(aixs_list), 'constant', constant_values=tuple(constant_list))
        #new_weight += np.random.normal(loc=0, scale=1e-6, size=new_weight.shape)
        print(old_weight.shape, weight.shape, new_weight.shape)
        if verification:
            old_input = np.random.rand(1,weight.shape[0])
            new_input = np.hstack([old_input, np.random.rand(1,new_width-weight.shape[0])])
            old_output = np.matmul(old_input,weight)
            new_output = np.matmul(new_input,new_weight)
            #print old_output-new_output
            assert np.mean(new_output-old_output) < 10-4, "obs wider error!"
            print("verification ok, output.mean:%d" % np.mean(new_output-old_output))
        return new_weight

    def wider_embedding(self, weight, bias, new_shape, verification=True):
        #assert weight.ndim == 4 or weight.ndim == 2, "check weight.ndim"       
        assert weight.ndim == 2, "check weight.ndim"       
        old_shape = weight.shape
        for i in range(len(new_shape)):
            assert new_shape[i] >= old_shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i], old_shape[i])
        
        new_weight = np.random.normal(loc = 0.0, scale = 5e-6, size= new_shape)
        #new_weight = np.zeros(new_shape)
        new_weight[:old_shape[0],:old_shape[1]] = weight
        new_bias = None
        if bias is not None:
            bias_shape = list(bias.shape)
            bias_shape[0] = new_shape[1]
            new_bias = np.zeros(bias_shape)
            new_bias[:bias.shape[0]] = bias
        return new_weight, new_bias
    
    def _wider_lstm(self, teacher_kernel, teacher_bias, teacher_w2_list, new_width, verification):
        teacher_kernel_split_0 = np.split(teacher_kernel, 2, axis=0)##inputs, h
        teacher_kernel_split_list = []
        for i in range(len(teacher_kernel_split_0)):
            teacher_kernel_split = np.split(teacher_kernel_split_0[i], 4, axis=1)##i,j,f,o
            teacher_kernel_split_list.append(teacher_kernel_split)
        teacher_bias_split = np.split(teacher_bias, 4, axis=0)
        student_kernel_split_list = []
        for i in range(len(teacher_kernel_split_list)):
            student_kernel_split_list.append([item.copy() for item in teacher_kernel_split_list[i]])
        student_bias_split = [item.copy() for item in teacher_bias_split]
        for i in range(len(teacher_kernel_split_list)):
            for j in range(len(teacher_kernel_split_list[i])):
                if i == 0:
                    tmp_shape = (teacher_kernel_split_list[i][j].shape[0], new_width-teacher_kernel_split_list[i][j].shape[1])
                    add_part = np.random.normal(loc = 0.0, scale = 5e-6, size= tmp_shape)
                    #add_part = np.zeros(tmp_shape)
                    student_kernel_split_list[i][j] = np.concatenate((student_kernel_split_list[i][j], add_part), axis=1)
                elif i == 1:
                    tmp_shape = (new_width - student_kernel_split_list[i][j].shape[0], student_kernel_split_list[i][j].shape[1])
                    add_part = np.random.normal(loc = 0.0, scale = 5e-6, size= tmp_shape)
                    #add_part = np.zeros(tmp_shape)
                    student_kernel_split_list[i][j] = np.concatenate((student_kernel_split_list[i][j], add_part), axis=0)
                    tmp_shape = (student_kernel_split_list[i][j].shape[0], new_width-student_kernel_split_list[i][j].shape[1])
                    add_part = np.random.normal(loc = 0.0, scale = 5e-6, size= tmp_shape)
                    #add_part = np.zeros(tmp_shape)
                    student_kernel_split_list[i][j] = np.concatenate((student_kernel_split_list[i][j], add_part), axis=1)
        for i in range(len(teacher_bias_split)):
            tmp_shape = (new_width-teacher_bias_split[i].shape[0])
            ###should be samll random values
            add_part = np.random.normal(loc = 0.0, scale = 5e-6, size= tmp_shape)
            #add_part = np.zeros(tmp_shape)
            student_bias_split[i] = np.concatenate((student_bias_split[i], add_part), axis=0)
        student_kernel_split_0 = []
        for i in range(len(student_kernel_split_list)):
            student_kernel_split_0.append(np.concatenate(tuple(student_kernel_split_list[i]), axis=1))
        student_kernel = np.concatenate(tuple(student_kernel_split_0), axis=0)     
        student_bias = np.concatenate(tuple(student_bias_split), axis=0) 
        student_w2_list = [teacher_w2.copy() for teacher_w2 in teacher_w2_list]
        for i in range(len(student_w2_list)):
            tmp_shape = (new_width-student_w2_list[i].shape[0], student_w2_list[i].shape[1])
            add_part = np.random.normal(loc = 0.0, scale = 5e-6, size= tmp_shape)
            #add_part = np.zeros(tmp_shape)
            student_w2_list[i] = np.concatenate((student_w2_list[i], add_part), axis=0)
            print (student_w2_list[i].shape)
        return student_kernel, student_bias, student_w2_list    
        
    def _wider_fc(self, teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        rand = np.random.randint(teacher_w1.shape[1], size=(new_width-teacher_w1.shape[1]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_weight = teacher_w1[:, teacher_index]
            new_weight = new_weight[:, np.newaxis]
            new_weight += np.random.normal(loc=0, scale=1e-7, size=new_weight.shape)
            student_w1 = np.concatenate((student_w1, new_weight), axis=1)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            new_weight = teacher_w2[teacher_index,:]*(1./factor)
            new_weight = new_weight[np.newaxis, :]
            new_weight += np.random.normal(loc=0, scale=1e-7, size=new_weight.shape)
            student_w2 = np.concatenate((student_w2, new_weight), axis=0)
            student_w2[teacher_index,:] = new_weight
        if verification:
            inputs = np.random.rand(1, teacher_w1.shape[0])
            ori1 = np.dot(inputs, teacher_w1) + teacher_b1
            ori2 = np.dot(ori1, teacher_w2)
            new1 = np.dot(inputs, student_w1) + student_b1
            new2 = np.dot(new1, student_w2)
            err = np.abs(np.sum(ori2-new2))
            assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return student_w1, student_b1, student_w2

    def _wider_fc_multi(self, teacher_w1, teacher_b1, teacher_w2_list, new_width, verification):
        rand = np.random.randint(teacher_w1.shape[1], size=(new_width-teacher_w1.shape[1]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2_list = []
        student_b1 = teacher_b1.copy()
        for teacher_w2 in teacher_w2_list:
            student_w2_list.append(teacher_w2.copy())
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_weight = teacher_w1[:, teacher_index]
            new_weight = new_weight[:, np.newaxis]
            new_weight += np.random.normal(loc=0, scale=1e-7, size=new_weight.shape)
            student_w1 = np.concatenate((student_w1, new_weight), axis=1)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            for idx in range(len(teacher_w2_list)):
                new_weight = teacher_w2_list[idx][teacher_index,:]*(1./factor)
                new_weight = new_weight[np.newaxis, :]
                new_weight += np.random.normal(loc=0, scale=1e-7, size=new_weight.shape)
                student_w2_list[idx] = np.concatenate((student_w2_list[idx], new_weight), axis=0)
                student_w2_list[idx][teacher_index,:] = new_weight
        if verification:
            inputs = np.random.rand(1, teacher_w1.shape[0])
            ori1 = np.dot(inputs, teacher_w1) + teacher_b1
            new1 = np.dot(inputs, student_w1) + student_b1
            for idx in range(len(teacher_w2_list)):
                ori2 = np.dot(ori1, teacher_w2_list[idx])
                new2 = np.dot(new1, student_w2_list[idx])
                err = np.abs(np.sum(ori2-new2))
                assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return student_w1, student_b1, student_w2_list
    
    def _wider_fc_rand(self, teacher_w1, teacher_b1, teacher_w2, new_width):
        size = new_width-teacher_w1.shape[1]
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(size):
            shape = teacher_w1[:,0].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight = new_weight[:, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=1)
            student_b1 = np.append(student_b1, 0.1)
        # next layer update (i+1)
        for i in range(size):
            shape = teacher_w2[0,:].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight = new_weight[np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight), axis=0)
        return student_w1, student_b1, student_w2

    def padding_for_adam(self, matrix, new_shape):
        aixs_list = []
        constant_list = []
        old_shape = matrix.shape
        # for i in range(matrix.ndim):
        #     assert new_shape[i] >= matrix.shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i], matrix.shape[i])
        #     tmp_axis = (0, new_shape[i] - matrix.shape[i])
        #     aixs_list.append(tmp_axis)
        #     constant_list.append((np.random.rand() - 0.5) * 0.5)
        # new_matrix = np.pad(matrix, tuple(aixs_list), 'constant', constant_values=tuple(constant_list))
        new_matrix = np.zeros(new_shape)
        for i in range(0, old_shape[0]):
            new_matrix[i:i+1,:] = matrix[i:i+1,:]
        for i in range(old_shape[0], new_shape[0]):
            new_matrix[i:i+1,:] = (np.random.rand(1, old_shape[1]) - 0.5) * 0.5 * 1e-8
        return new_matrix

    def padding_for_adam2(self, old_matrix, new_shape, start, end):
        matrix = old_matrix[start:end]
        aixs_list = []
        constant_list = []
        old_shape = matrix.shape
        for i in range(matrix.ndim):
            assert new_shape[i] >= matrix.shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i], matrix.shape[i])
            tmp_axis = (0, new_shape[i] - matrix.shape[i])
            aixs_list.append(tmp_axis)
            constant_list.append(0.0)
        new_matrix = np.pad(matrix, tuple(aixs_list), 'constant', constant_values=tuple(constant_list))
        print(old_matrix.shape, matrix.shape, new_matrix.shape)
        return new_matrix
        
    def padding_for_lstm_adam(self, matrix, new_shape):
        if matrix.ndim == 2:
            new_width = new_shape[1]/4
            matrix_split = np.split(matrix, 2, axis=0)
            matrix_split_list = []
            new_matrix_split_list = []
            for i in range(len(matrix_split)):
                matrix_split_list.append(np.split(matrix_split[i], 4, axis=1))
                new_matrix_split_list.append([item.copy() for item in matrix_split_list[i]])
            for i in range(len(matrix_split_list)):
                for j in range(len(matrix_split_list[i])):
                    if i == 0:
                        tmp_shape = (new_matrix_split_list[i][j].shape[0], new_width-new_matrix_split_list[i][j].shape[1])
                        add_part = np.zeros(tmp_shape)
                        new_matrix_split_list[i][j] = np.concatenate((new_matrix_split_list[i][j], add_part), axis=1)
                    elif i == 1:
                        tmp_shape = (new_width - new_matrix_split_list[i][j].shape[0], new_matrix_split_list[i][j].shape[1])
                        add_part = np.zeros(tmp_shape)
                        new_matrix_split_list[i][j] = np.concatenate((new_matrix_split_list[i][j], add_part), axis=0)
                        tmp_shape = (new_matrix_split_list[i][j].shape[0], new_width-new_matrix_split_list[i][j].shape[1])
                        add_part = np.zeros(tmp_shape)
                        new_matrix_split_list[i][j] = np.concatenate((new_matrix_split_list[i][j], add_part), axis=1)
            new_matrix_split = []
            for i in range(len(matrix_split_list)):
                new_matrix_split.append(np.concatenate(tuple(new_matrix_split_list[i]), axis=1))
            new_matrix = np.concatenate(tuple(new_matrix_split), axis=0)
        else:
            matrix_split = np.split(matrix, 4, axis=0)
            new_matrix_split = []
            for s in range(len(matrix_split)):
                aixs_list = []
                constant_list = []
                for i in range(matrix_split[s].ndim):
                    if i == matrix.ndim-1:
                        assert new_shape[i]/4 >= matrix_split[s].shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i]/4, matrix_split[s].shape[i])
                        tmp_axis = (0, new_shape[i]/4 - matrix_split[s].shape[i])
                    else:
                        assert new_shape[i] >= matrix_split[s].shape[i], "new_shape %d must be larger than old_shape %d" %(new_shape[i], matrix_split[s].shape[i])
                        tmp_axis = (0, new_shape[i] - matrix_split[s].shape[i])
                    aixs_list.append(tmp_axis)
                    constant_list.append(0.0)
                new_matrix_split.append(np.pad(matrix_split[i], tuple(aixs_list), 'constant', constant_values=tuple(constant_list)))
            new_matrix = np.concatenate(tuple(new_matrix_split), axis=matrix.ndim-1)
        return new_matrix

if __name__ == '__main__':
    """ Net2Net Class Test """
    obj = Net2Net()

    w1 = np.random.rand(100, 50)
    obj.deeper(w1)
    print ('Succeed: Net2Deeper (fc)')
    
    w1 = np.random.rand(3,3,16,32)
    obj.deeper(w1)
    print ('Succeed: Net2Deeper (conv)')
    
    w1 = np.random.rand(100, 50)
    b1 = np.random.rand(50,1)
    w2 = np.random.rand(50, 10)
    obj.wider(w1, b1, w2, 70)
    print ('Succeed: Net2Wider (fc)')

    w1 = np.random.rand(3,3,16,32)
    b1 = np.random.rand(32)
    w2 = np.random.rand(3,3,32,64)
    obj.wider(w1, b1, w2, 48)
    print ('Succeed: Net2Wider (conv)')
