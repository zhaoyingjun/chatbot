<template>
  <div class="box">
    <Header></Header>
    <div class="chat">
      <div class="chat-title">
        <h1>大 头</h1>
        <h2>想要挨一jio嘛</h2>
        <figure class="avatar">
          <img src="static/image/toubao.jpg" />
        </figure>
      </div>
      <div class="messages">
        <div class="messages-content">
          <Message @sendMsg="sendMsg" :resMsg="resMsg"></Message>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Header from "./common/Header";
import $ from 'jquery';
import qs from 'qs';
import Message from './common/message.vue';

export default {
  name: "Chat",
  components: {
    Header,
    Message
  },
  data() {
    return {
      resMsg: {}
    }
  },

  created() {
  },

  methods: {
    check_user_login() {
      // 获取用户的登录状态
      this.token = sessionStorage.user_token || localStorage.user_token;
      return this.token;
    },

    setDate() {
      let d = new Date();
      let m;
      if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
      }
    },
    // 接收子组件的传参
    sendMsg(msg) {
      if (!msg) return this.$message.warning('请输入内容')
      this.interact(msg)
    },
    // 消息请求
    interact(message) {
      if (this.check_user_login()) {
        const data = {
          question: message,
          user: sessionStorage.user_name || localStorage.user_name,
        }
        this.$axios.post(`${this.$settings.HOST}/user/chat/`,
          qs.stringify(data),

        ).then(response => {
          console.log(response);
          this.resMsg = response.data
        }).catch(error => {
          console.log(error);
        });
      } else {
        this.$axios.post(`${this.$settings.HOST}/user/chat-tour/`, {
          question: message,
        }).then((res)=> {
          console.log(res);
          this.resMsg = res.data

        }).catch(error => {
          console.log(error);
        });
      }
    },
  },
}
</script>

<style scoped>
@import "../../src/style/css/normalize.css";
@import "../style/css/chat.css";

.box {
  position: fixed;
  background-color: #d3dce6;
  height: 100%;
  width: 100%;
}
</style>
