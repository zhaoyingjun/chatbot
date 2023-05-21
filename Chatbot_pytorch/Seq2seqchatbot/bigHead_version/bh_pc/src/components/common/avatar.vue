<template>
    <div>
        <el-upload class="upload-demo" 
          action=""   
          ref="uploadimport" 
          :on-change="handleUpload"
          :auto-upload="false"
          :limit="1" 
          :show-file-list="false" >
          <el-button size="small" type="primary" circle></el-button>
        </el-upload>
        <!-- <el-upload
  class="upload-demo"
  ref="upload"
  action="https://jsonplaceholder.typicode.com/posts/"
  :on-preview="handlePreview"
  :limit="1" 
  :on-change="handleUpload"
  :auto-upload="false">
</el-upload> -->
    </div>
</template>
<script>
import qs from 'qs'
  export default {
    data() {
      return {
        circleUrl:'',
        username:localStorage.user_name || sessionStorage.user_name,
        fileList: [{name: 'food.jpeg', url: '../../static/image/default_avatar.png'}]
      };
    },
    methods: {
        handleUpload(val,fileList){
          console.log(val,fileList);
          const formDatas = new FormData()
          formDatas.append("avatar",val.raw)
          this.$axios
        // .put(`${this.$settings.HOST}/user/person/${this.username}/`,JSON.stringify(data))
          .put(`${this.$settings.HOST}/user/person/${this.username}/`, formDatas,{
                'Content-type' : 'multipart/form-data'
            })
          .then(res=>{
              console.log(res);
              this.$emit('updateAvatar',res.data.avatar)
          })
        },
        handleSuccess(res,file,fileList){
          console.log(res,file,fileList);
        },
        handleProgress(e,f,fl){
          console.log(e,f,fl);
          // this.handleUpload(f)
        },
        uploadBefore(e){
            console.log(e);
            this.handleUpload(e)
        },
   
      handlePreview(file) {
        console.log(file);
      },
      handleExceed(files, fileList) {
        console.log(files);
        this.$message.warning(`当前限制选择 3 个文件，本次选择了 ${files.length} 个文件，共选择了 ${files.length + fileList.length} 个文件`);
      },
    }
  }
</script>
<style lang="less" scoped>
.el-button--primary{
    width: 120px;
    height: 120px;
    background-color: transparent;
    border: none;
}
</style>