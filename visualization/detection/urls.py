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
    url(r'^api/test_model$', 
        views.api_test_model,
        name='api_test_model'),
    url(r'^confirm/$', 
        views.confirm,
        name='confirm'),
    url(r'^media/(.*)$',
        views.download_file,
        name='return_file')
]

