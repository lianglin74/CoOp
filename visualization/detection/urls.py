from django.conf.urls import url
from . import views

app_name = 'detection'

urlpatterns = [
    url(r'taxonomy_validation_result/$',
        views.validate_taxonomy,
        name='taxonomy_validation_result'),
    url(r'^view_image/$', 
        views.view_image, 
        name='view_image'),
    url(r'^view_result/$', 
        views.view_result, 
        name='view_result'),
    url(r'^view_compare/$', 
        views.view_compare, 
        name='view_compare'),
    url(r'^view_compare_all/$', 
        views.view_compare_all,
        name='view_compare_all'),
    url(r'^view_compare_test/$', 
        views.view_compare_test, 
        name='view_compare_test'),
    url(r'^taxonomy_verification/$',
        views.input_taxonomy,
        name='taxonomy_verification'),
    url(r'^view_tree/$', 
        views.view_tree, 
        name='view_tree'),
    url(r'^view_tree_test/$', 
        views.view_tree_test, 
        name='view_tree_test'),
    url(r'^verify_data/$',
        views.verify_data,
        name='verify_data'),
    url(r'^view_video/$',
        views.view_video,
        name='view_video'),
    url(r'^view_video_2/$',
        views.view_video_2,
        name='view_video_2'),
    url(r'^view_video_test/$',
        views.view_video_test,
        name='view_video_test'),
    url(r'^video/upload$',
        views.upload_video,
        name='upload_video'),
#     $env:AZCOPY_CRED_TYPE = "Anonymous";
# ./azcopy.exe copy "I:\ChinaMobile\Video\CBA\5102215894_5004648487_92.mp4" "https://vigstandard.blob.core.windows.net/data/video/5102215894_5004648487_92.mp4?se=2019-07-11T18%3A25%3A53Z&sp=rwl&sv=2018-03-28&sr=c&sig=I5pxsFjQ7cV5s7O2%2Fm60yEXoPQcDZ6f5cYsyywUECq8%3D" --overwrite=false --follow-symlinks --recursive --from-to=LocalBlob --blob-type=BlockBlob --put-md5;
# $env:AZCOPY_CRED_TYPE = "";

    # if no parameter is given -> show all the exps
    # if full_expid is given -> show all the prediction results, i.e. *.predict
    # file
    # if full_expid is given and file name of predict file is given -> show the
    # prediction results. 
    # old scenario: full_expid is not specified. 
    url(r'^view_model/$', 
        views.view_model,
        name='view_model'),
    url(r'^edit_model_label/$', 
        views.edit_model_label,
        name='edit_model_label'),
    url(r'^test_model/$', 
        views.test_model,
        name='test_model'),
    url(r'^confirm/$', 
        views.confirm,
        name='confirm'),
    url(r'^media/(.*)$',
        views.download_file,
        name='return_file')
]

