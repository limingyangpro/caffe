// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   read_lmdb --database=/path/to/lmdb/folder [FLAGS]

#include <iostream>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "opencv/cv.hpp"
#include "opencv/highgui.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(database, "",
        "The path of database");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Show images stored inside the leveldb/lmdb\n"
        "Usage:\n"
        "    read_lmdb --database=/path/to/lmdb/folder [FLAGS]\n"

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string source = FLAGS_database;
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(source, db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());
  //EXPECT_TRUE(cursor->valid());

  while (cursor->valid()){
     string key = cursor->key();
     Datum datum;
     datum.ParseFromString(cursor->value());
     int channels = datum.channels();
     int height = datum.height();
     int width = datum.width();
     int mattype;
     LOG(INFO)<<key<<" : "<<height<<"x"<<width<<" in " << channels << " channels."<<std::endl;

     if (channels == 1){
        mattype = CV_8UC1;
     } else {
        mattype = CV_8UC3;
     }

     cv::Mat img(height, width, mattype);
     if (!datum.encoded()){
        img = DatumToCVMat(datum);
     } else {
        img = DecodeDatumToCVMat(datum, true);
     }
     cv::imshow("Loaded", img);
     cv::waitKey(-1);
     cursor->Next();
  }

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
