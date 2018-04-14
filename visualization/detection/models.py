# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.

class File(models.Model):
    file = models.FileField(upload_to = './')
    name = models.CharField(max_length = 255)
