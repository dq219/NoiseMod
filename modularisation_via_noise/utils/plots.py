import matplotlib.pyplot as plt
import numpy as np 
import matplotlib as mpl

def plot_test(model, dataset, device, morph, folder, file_name, file_format = 'pdf', figsize = (4,3), mixed = False, **kwargs):

	inputs, outputs = dataset.get_all_data()
	inputs, outputs = inputs.to(device), outputs.to(device)
	model.eval()
	pred = model(inputs)[0].detach()
	encode_num = len(inputs[0])
	decode_num = len(outputs[0])

	if mixed == False:
		xx = np.linspace(-3, 3, 100)
		yy = [morph([x] * encode_num) for x in xx]

		yy = np.array(yy)

		fig, axes = plt.subplots(1, decode_num, figsize = (9, 2), sharex = True, sharey = True)
		for i in range(decode_num):
		    if decode_num == 1:
		        axes.scatter(inputs[:, i].cpu().numpy(), pred[:, i].cpu().numpy(), s = 2)
		        axes.plot(xx, yy[:, i], color = 'black')
		        #axes.set_aspect(1)
		    else:
		        axes[i].scatter(inputs[:, i].cpu().numpy(), pred[:, i].cpu().numpy(), s = 2)
		        axes[i].plot(xx, yy[:, i], color = 'black')
		        #axes[i].set_aspect(1)

		plt.savefig(f'{folder}/{file_name}.{file_format}', **kwargs)
		plt.close()
	else:
		fig, axes = plt.subplots(1, decode_num, figsize = (9, 2), sharex = True, sharey = True)
		for i in range(decode_num):
		    if decode_num == 1:
		        axes.scatter(outputs[:, i].cpu().numpy(), pred[:, i].cpu().numpy(), s = 2)
		        axes.set_aspect(1)
		    else:
		        axes[i].scatter(outputs[:, i].cpu().numpy(), pred[:, i].cpu().numpy(), s = 2)
		        axes[i].set_aspect(1)

		plt.savefig(f'{folder}/{file_name}.{file_format}', **kwargs)
		plt.close()


def plot_loss(losses, folder, file_name, file_format = 'pdf', figsize = (4,3), **kwargs):

	plt.figure(figsize = figsize)
	labels = ['data loss', 'validation', 'L1', 'L2', 'LWA', 'LA', 'L12']
	for i in range(losses.shape[-1]):
		if losses[0, i] > 0:
			plt.plot(losses[:, i], label = labels[i])
	plt.yscale('log')
	plt.ylim(1e-6, 1e3)
	plt.legend()
	plt.savefig(f'{folder}/{file_name}.{file_format}', **kwargs)
	plt.close()

def plot_spectrum(EIG, folder, file_name, file_format = 'pdf', figsize = (4,3), **kwargs):

	plt.figure(figsize = figsize)

	for i in range(len(EIG)):

	    color = mpl.colormaps.get_cmap('plasma')(i/len(EIG))
	    
	    eigval = EIG[i][0]
	    eigidx = EIG[i][1]
	    eigmsk = eigidx < 11

	    plt.scatter(eigidx[eigmsk], eigval[eigmsk], color = color, s = 10)
	    plt.plot(eigidx[eigmsk], eigval[eigmsk], color = color, zorder = -10, label = f'layer {i}')

	plt.legend()
	plt.xlabel('eigenmode index')
	plt.ylabel('eigenvalue')
	plt.xlim(0, 11)
	plt.ylim(-0.2,1.2)
	plt.tight_layout()
	plt.savefig(f'{folder}/{file_name}.{file_format}', **kwargs)
	plt.close()

def plot_connectivity(WMATS, EIG, IDX, k, cmap, folder, file_name, file_format = 'pdf', figsize = (4,3), **kwargs):

	fig, axes = plt.subplots(1, len(WMATS), figsize = (12, 4))
	for i in range(len(WMATS)):

	    [sorted_indices, inxBounds] = IDX[i]
	    [sorted_indices_new, inxBounds_new] = IDX[i + 1]
	    
	    mat = WMATS[i][sorted_indices, :]
	    (dim_in, dim_out) = mat.shape

	    for j in range(1, k):
	        axes[i].axhline(dim_in - inxBounds[j], color = 'black')
	        
	    mMin = np.min(mat)
	    mMax = np.max(mat)
	    vm = np.max(np.abs([mMin, mMax])) * 1.1
	    print(f'min - max : {mMin:.3f} - {mMax:.3f}')
	    
	    axes[i].imshow(mat[:, sorted_indices_new],
	               interpolation='none', rasterized=True,
	               cmap = cmap, vmin = -vm, vmax = vm, extent = (0, dim_out, 0, dim_in))
	    
	    for j in range(1, k):
	        axes[i].axvline(inxBounds_new[j], color = 'black')
	        
	    #S_reordered = S[sorted_indices, :][:, sorted_indices]
	    #fig, axes = plt.subplots(1, 2, figsize = (4, 2))
	    #axes[0].imshow(S)
	    #axes[1].imshow(S_reordered)
	    
	plt.savefig(f'{folder}/{file_name}.{file_format}', **kwargs)
	plt.close()





