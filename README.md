Super-Resolution GAN
======================

:star: Star this project on GitHub — it helps!

[Super-Resolution GAN](https://arxiv.org/abs/1609.04802) is one of the best examples in GAN's application. It allows
to imporove the quality of input low-resolution image. As **Generator** this GAN uses
**SRResNet** (Super-Resolution Residual Network). In order to provide better quality of
the output image the network is trained on three different loss functions.


## Table of content

- [Evaluation](#eval)
    - [TER](#typo3-extension-repository)
- [Training](#train)
    - [Database](#database)
    - [WarmUp of Generator](#warmup)
    - [Training with Discriminator](#train)
- [License](#license)
- [Links](#links)

## Evaluation

This document is for the latest Aimeos TYPO3 **20.10 release and later**.

- LTS release: 20.10 (TYPO3 9/10 LTS)
- Beta release: 2021.01 (TYPO3 9/10 LTS)

### Training

#### Database

If you want to install Aimeos into your existing TYPO3 installation, the [Aimeos extension from the TER](https://typo3.org/extensions/repository/view/aimeos) is recommended. You can download and install it directly from the Extension Manager of your TYPO3 instance.

![](images/)

For new TYPO3 installations, there's a 1-click [Aimeos distribution](https://typo3.org/extensions/repository/view/aimeos_dist) available too. Choose the Aimeos distribution from the list of available distributions in the Extension Manager and you will get a completely set up shop system including demo data for a quick start.

## License

The Aimeos TYPO3 extension is licensed under the terms of the GPL Open Source
license and is available for free.

## Links

* [Web site](https://aimeos.org/integrations/typo3-shop-extension/)
* [Documentation](https://aimeos.org/docs/TYPO3)
* [Forum](https://aimeos.org/help/typo3-extension-f16/)
* [Issue tracker](https://github.com/aimeos/aimeos-typo3/issues)
* [Source code](https://github.com/aimeos/aimeos-typo3)
