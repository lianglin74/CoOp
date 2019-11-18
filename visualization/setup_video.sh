sudo su -
 
apt-get update
apt-get install -y apache2
apt-get install -y apache2-dev

mkdir -p /tmp/setup_video
cd /tmp/setup_video
wget http://h264.code-shop.com/download/apache_mod_h264_streaming-2.2.7.tar.gz
tar -zxvf apache_mod_h264_streaming-2.2.7.tar.gz
cd mod_h264_streaming-2.2.7/
./configure --with-apxs='/usr/bin/apxs2'
 
make
make install

sed -i -e '$a\
LoadModule h264_streaming_module /usr/lib/apache2/modules/mod_h264_streaming.so\
AddHandler h264-streaming.extensions .mp4' /etc/apache2/apache2.conf

mkdir -p /var/www/html/mp4
ln -s /raid/data/video/CBA/CBA_demo /var/www/html/mp4/
