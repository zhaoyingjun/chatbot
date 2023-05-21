<template>
  <div class="box">
    <div class="register" >
      <div class="register_box">
        <div class="register-title">注册大头AI</div>
        <div class="inp">
          <input v-model="mobile" type="text" @blur="checkMobile" placeholder="手机号码" class="user">
          <input v-model="password" type="password" placeholder="登录密码" class="user">
          <div class="sms-box">
            <input v-model="sms_code" maxlength="6" type="text" placeholder="输入验证码" class="user">
            <div class="sms-btn" @click="smsHandler">{{ sms_text }}</div>
          </div>
          <button class="register_btn" @click="registerHandler">注册</button>
          <p class="go_login">已有账号
            <router-link to="/user/login">直接登录</router-link>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Register',
  data() {
    return {
      sms_code: "",
      mobile: "",
      password: "",
      is_send_sms: false,
      sms_text: "点击发送短信",

    }
  },
  created() {
  },
  methods: {
    checkMobile() {
      this.$axios
        .get(`${this.$settings.HOST}/user/mobile/${this.mobile}/`)
        .catch(error => {this.$message.error(error.response.data.message);
      });
    },
    registerHandler() {
      this.$axios
        .post(`${this.$settings.HOST}/user/`, {
        mobile: this.mobile,
        sms_code: this.sms_code,
        password: this.password,
      }).then(response => {
        console.log(response.data);
        localStorage.removeItem("user_token");
        localStorage.removeItem("user_id");
        localStorage.removeItem("user_name");
        sessionStorage.user_token = response.data.token;
        sessionStorage.user_id = response.data.id;
        sessionStorage.user_name = response.data.username;

        let self = this;
        this.$alert("注册成功!", "路飞学城", {
          callback() {
            self.$router.push("/");
          }
        });

      }).catch(error => {
        console.log(error.response)
        let data = error.response.data;
        let message = "";
        for (let key in data) {
          message = data[key][0];
        }
        this.$message.error(message);
      });
    },
    smsHandler() {
      if (!/1[3-9]\d{9}/.test(this.mobile)) {
        this.$message.error("手机号码格式不正确！");
        return false;
      }

      if (this.is_send_sms) {
        this.$message.error("当前手机号已经在60秒内发送过短信，请不要频繁发送！");
        return false;
      }

      this.$axios
        .get(`${this.$settings.HOST}/user/sms/${this.mobile}/`)
        .then(response => {
        console.log(response.data);
        this.is_send_sms = true;
        let interval_time = 60;
        let timer = setInterval(() => {
          if (interval_time <= 1) {
            clearInterval(timer);
            this.is_send_sms = false;
            this.sms_text = "点击发送短信";
          } else {
            interval_time--;
            this.sms_text = `${interval_time}秒后重新点击发送`;
          }
        }, 1000)
      }).catch(error => {
        this.$message.error(error.response.data.message);
      });
    }
  }
};
</script>

<style scoped>
.box {
  width: 100%;
  height: 100%;
  position: fixed;
  background-color:#b5c6c6;
}

.box .register {
  position: absolute;
  top: 100px;
  left: 0;
  right: 0;
  margin: 0 auto;

}

.register .register-title {
  width: 100%;
  font-size: 24px;
  text-align: center;
  padding-top: 30px;
  padding-bottom: 30px;
  color: #4a4a4a;
  letter-spacing: .39px;
}

.register-title img {
  width: 190px;
  height: auto;
}

.register-title p {
  font-family: PingFangSC-Regular;
  font-size: 18px;
  color: #fff;
  letter-spacing: .29px;
  padding-top: 10px;
  padding-bottom: 50px;
}

.register_box {
  width: 400px;
  height: auto;
  background: #fff;
  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, .5);
  border-radius: 4px;
  margin: 0 auto;
  padding-bottom: 40px;
}

.register_box .title {
  font-size: 20px;
  color: #9b9b9b;
  letter-spacing: .32px;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  justify-content: space-around;
  padding: 50px 60px 0 60px;
  margin-bottom: 20px;
  cursor: pointer;
}

.register_box .title span:nth-of-type(1) {
  color: #4a4a4a;
  border-bottom: 2px solid #84cc39;
}

.inp {
  width: 350px;
  margin: 0 auto;
}

.inp input {
  border: 0;
  outline: 0;
  width: 100%;
  height: 45px;
  border-radius: 4px;
  border: 1px solid #d9d9d9;
  text-indent: 20px;
  font-size: 14px;
  background: #fff !important;
}

.inp input.user {
  margin-bottom: 16px;
}

.inp .remember {
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: relative;
  margin-top: 10px;
}

.inp .remember p:first-of-type {
  font-size: 12px;
  color: #4a4a4a;
  letter-spacing: .19px;
  margin-left: 22px;
  display: -ms-flexbox;
  display: flex;
  -ms-flex-align: center;
  align-items: center;
}

.inp .remember p:nth-of-type(2) {
  font-size: 14px;
  color: #9b9b9b;
  letter-spacing: .19px;
  cursor: pointer;
}

.inp .remember input {
  outline: 0;
  width: 30px;
  height: 45px;
  border-radius: 4px;
  border: 1px solid #d9d9d9;
  text-indent: 20px;
  font-size: 14px;
  background: #fff !important;
}

.inp .remember p span {
  display: inline-block;
  font-size: 12px;
  width: 100px;
}

#geetest {
  margin-top: 20px;
}

.register_btn {
  width: 100%;
  height: 45px;
  background: #88c0d0;;
  border-radius: 5px;
  font-size: 16px;
  color: #fff;
  letter-spacing: .26px;
  margin-top: 30px;
}

.inp .go_login {
  text-align: center;
  font-size: 14px;
  color: #9b9b9b;
  letter-spacing: .26px;
  padding-top: 20px;
}

.inp .go_login span {
  color: #84cc39;
  cursor: pointer;
}

.sms-box {
  position: relative;
}

.sms-btn {
  font-size: 14px;
  color: #9cacb6;
  letter-spacing: .26px;
  position: absolute;
  right: 16px;
  top: 12px;
  cursor: pointer;
  overflow: hidden;
  background: #fff;
  border-left: 1px solid #484848;
  padding-left: 16px;
  padding-bottom: 4px;
}
</style>
