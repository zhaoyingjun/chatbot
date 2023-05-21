<template>
  <div class="box">
    <Header></Header>
    <div class="conteiner_box">
      <div class="left-box">
        <el-row class="row-bg" >
          <el-col :span="6" v-for="item in chatData" :key="item.id" >
            <div class="grid-content bg-purple" @click="chatDetail(item.created_day)">
              <h5>{{item.created_day}}</h5>
              <el-divider content-position="center"></el-divider>
              <div class="chatMsg">
                <span>{{  $moment(item.created_time).format("HH:mm") }}</span>
                <p><span>{{ nickname }}:</span> {{ item.question }}</p>
                <p> &nbsp </p>
                <span>{{ $moment(item.created_time).format("HH:mm") }}</span>
                <p><span>大头:</span> {{ item.answer }}</p>
                <p> &nbsp </p>
                <p><span>......</span></p>
              </div>
            </div>
          </el-col>
        </el-row>
      </div>
      <div class="right_box">
        <el-row>
          <el-card :body-style="{ padding: '0px' }">
            <div class="image_box">
              <div class="image_container">

                <Avatar class="userImage" ref="userImage" @updateAvatar="updateAvatar"></Avatar>
                <img :src="avatar" class="image" v-if="avatar">
                <img src="../../static/image/default_avatar.png" class="image" v-else>
              </div>
              <div class="bottom clearfix" >
                <el-button type="text" class="button" @click="editevent">{{ nickname }}</el-button>
              </div>
              <div class="editBtn">
                <el-button type="primary" icon="el-icon-edit" @click="editevent" round>修改昵称</el-button>
              </div>
            </div>
          </el-card>
        </el-row>
      </div>
    </div>
<EditName :edit-data="editData" @editData="edit"></EditName>
  </div>
</template>

<script>
import qs from 'qs'
import Header from "./common/Header"
import EditName from "./common/EditName.vue"
import Avatar from "./common/avatar.vue"
import { markRaw } from 'vue'
export default {
  name: "User",
  data() {
    return {
      nickname: localStorage.nickname || sessionStorage.nickname,
      avatar: localStorage.avatar || sessionStorage.avatar,
      editData:{},
      username:localStorage.user_name || sessionStorage.user_name,
      chatData:[]
    }
  },
  created() {
    this.chattingRecords()
  },
  methods: {
    updateNickname(val) {
      const data = {
        nickname: val.name,
      }
      this.$axios
        .put(`${this.$settings.HOST}/user/person/${this.username}/`, qs.stringify(data)).then(res=>{
          console.log(res);
          if(res.status==200){
            this.nickname = res.data.nickname
            localStorage.nickname = res.data.nickname
          sessionStorage.nickname = res.data.nickname
            this.$notify.success('用户名修改成功！')
          }
        })
        .catch(error => {
          this.$notify.error('用户名修改失败！')
          console.log(error);
        });
    },
    updateAvatar(e){
      this.avatar = e
      localStorage.avatar = e
      sessionStorage.avatar = e
    },
    editevent(){
      this.editData = {
        status:true,
        username:localStorage.nickname ? localStorage.nickname : '用户名',

      }
    },
    edit(val){
      console.log(val);
      this.updateNickname(val)
    },
    chattingRecords(){
      if( localStorage.user_name || sessionStorage.user_name){
        const username = localStorage.user_name || sessionStorage.user_name
        this.$axios.get(`${this.$settings.HOST}/user/chatlog/${username}/`).then(res=>{
          const map = new Map()
          for(let item of res.data){
            map.set(item.created_day,item)
          }
          this.chatData = [...map.values()]
          console.log(this.chatData);
        })
      }
    },
    chatDetail(day){
      const username = localStorage.user_name || sessionStorage.user_name
      this.$axios.get(`${this.$settings.HOST}/user/chatlog/${username}/${day}`).then(res=>{
        console.log(res);
        this.$router.push({
          name:'Chat',
          params:{
            chatMsg:res.data,
            status:true
          }
        })
      })
    }
  },
  components: {
    Header,
    EditName,
    Avatar
  }
}
</script>

<style lang="less" scoped>
.conteiner_box {
  display: flex;
}
.image_container{
  position: relative;
  .userImage{
    position: absolute;
    left: 33%;
  }
}
.left-box {
  background-color: #b4c0d0;
  overflow: hidden;
  overflow-y: scroll;
  width: 900px;
  height: 590px;
  padding: 5px;


  .el-row {
    top: 0 !important;
    left: 0 !important;
    margin: 26px 60px !important;
  }
  .el-col-6{
    margin: 14px 30px;
  }

}
::-webkit-scrollbar {
  width:12px;
}

::-webkit-scrollbar-track {
  -webkit-box-shadow:inset006pxrgba(0,0,0,0.3);
  border-radius:10px;
}

::-webkit-scrollbar-thumb {
  border-radius:10px;
  background:#4b555d;
  -webkit-box-shadow:inset006pxrgba(0,0,0,0.5);
}

.right_box {
  flex: 1;

  .el-row {
    top: 0;
    left: 0;
    margin: 0;

    .el-card{
      background: #d3dce6;
      border: none;
      padding: 17px;
    border-radius: unset;
    }
  }
}

.image_box {
  height: 490px;

  img {
    display: block;
    margin: 75px auto;
    border-radius: 50%;
    width: 120px;
  }
  .bottom{
    text-align: center;
    .el-button--text{
      color: #797c83;
    }
  }
  .editBtn{
    text-align: center;
    .el-button--primary{
      background-color: #49565d;
      border-color: #49565d;
    }
  }
}

.avatar {
  position: relative;
  top: 100px;
  left: 100px;
}

.avatar-input {
  opacity: 0;
  cursor: pointer;
  position: absolute;
  left: 20px;

  height: 150px;

}

.name-box {
  position: absolute;
  top: 300px;
  right: 28px;
}

.prompt {
  position: absolute;
  right: 100px;
  color: #47555e;

}

.el-input {
  position: absolute;
  right: 38px;
  width: 200px;
  top: 50px;
}

.icon {
  position: absolute;
  top: 58px;
  right: 8px;
  display: block;
  color: #47555e;
}

.el-icon-s-check {
  display: none;
}

.el-row {
  position: relative;
  top: 15px;
  left: 58px;

  margin: 30px;
}

.bg-purple {
  background: #d3dce6;
  height: 230px;
}

.grid-content {
  padding: 5px 10px;
  border-radius: 20px;
  cursor: pointer;
  h5{
    padding-top: 15px;
    color: #48555e;
    margin: 0;
  }
  .el-divider{
    background-color: #aec3d0;
    margin: 15px 0;
  }
  .chatMsg{
    span{
      font-size: 15px;
      color: #797c83;
    }
    p{
      color: #49565d;
      font-size: 14.5px;
    }
  }
}
</style>
