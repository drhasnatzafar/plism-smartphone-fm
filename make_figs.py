import pandas as pd, matplotlib.pyplot as plt, numpy as np, sys
R = sys.argv[1]
N = {'uni2h':'UNI2-h','gigapath':'Prov-GigaPath','hoptimus0':'H-optimus-0','uni':'UNI','virchow2':'Virchow2',
     'phikon2':'Phikon-v2','dinov2_l':'DINOv2-L','conch':'CONCH','vitb16_in':'ViT-B/16 (ImageNet)'}
P = {'iphone13':'iPhone 13 mini','redmi':'Redmi','moto':'Moto','iphone6':'iPhone 6','galaxy':'Galaxy','itel':'itel'}
m = pd.read_csv(f'{R}/classification_main.csv')
m['lo'] = m['drop [95% CI]'].str.extract(r'\[([\d.]+),').astype(float); m['hi'] = m['drop [95% CI]'].str.extract(r', ([\d.]+)\]').astype(float)
m['d'] = m['drop [95% CI]'].str.split(' ').str[0].astype(float)
lr = m[m.probe=='lr'].set_index('model'); kn = m[m.probe=='knn'].set_index('model')
order = lr.sort_values('d').index[::-1]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={'width_ratios':[1, 1.25]})
y = np.arange(len(order))
for off, t, lab, c in ((0.15, lr, 'linear probe', 'C0'), (-0.15, kn, 'kNN probe', 'C1')):
    t = t.loc[order]
    ax[0].errorbar(t.d, y + off, xerr=[t.d - t.lo, t.hi - t.d], fmt='o', ms=4, capsize=2, color=c, label=lab)
ax[0].set_yticks(y); ax[0].set_yticklabels([N[k] for k in order]); ax[0].axvline(0, c='k', lw=.5)
ax[0].set_xlabel('balanced-accuracy drop (scanner $-$ phone)'); ax[0].legend(frameon=False, fontsize=8, loc='upper right'); ax[0].set_title('(a) scanner$\\to$phone drop', fontsize=10)
pp = pd.read_csv(f'{R}/per_phone_drop.csv').set_index('model').loc[order[::-1]]
cols = pp.mean().sort_values().index; pp = pp[cols]
im = ax[1].imshow(pp.values, cmap='viridis', aspect='auto')
ax[1].set_yticks(range(len(pp))); ax[1].set_yticklabels([N[k] for k in pp.index], fontsize=8)
ax[1].set_xticks(range(len(cols))); ax[1].set_xticklabels([P[c] for c in cols], rotation=30, ha='right', fontsize=8)
for i in range(pp.shape[0]):
    for j in range(pp.shape[1]):
        v = pp.values[i, j]; ax[1].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5, color='w' if v < .2 else 'k')
fig.colorbar(im, ax=ax[1], label='BA drop (linear probe)'); ax[1].set_title('(b) drop per phone', fontsize=10)
fig.tight_layout(); fig.savefig(f'{R}/figures/fig_main.pdf'); fig.savefig(f'{R}/figures/fig_main.png', dpi=150)
