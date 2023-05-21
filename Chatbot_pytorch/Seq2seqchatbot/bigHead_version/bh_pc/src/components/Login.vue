<template>
	<div class="login box">
		<div class="login">
			<div class="login_box">
				<div class="title">
          <span>登录账号</span>
				</div>
				<div class="inp" v-if="login_type==0">
					<input v-model = "username" type="text" placeholder="手机号" class="user">
					<input v-model = "password" type="password" name="" class="pwd" placeholder="密码">
					<div id="geetest1"></div>
					<div class="remember">
						<p>
							<input type="checkbox" class="no" name="a" v-model="remember"/>
							<span>记住密码</span>
						</p>
						<p>忘记密码</p>
					</div>
					<button class="login_btn" @click="loginHandler">登录</button>
					<p class="go_login" >没有账号 <router-link to="/user/reg">立即注册</router-link></p>
				</div>
				<div class="inp" v-show="login_type==1">
					<input v-model = "username" type="text" placeholder="手机号码" class="user">
					<input v-model = "password"  type="text" class="pwd" placeholder="短信验证码">
          <button id="get_code" style="margin: 20px 0 20px 0;">获取验证码</button>
					<button class="login_btn" @click="loginHandler">登录</button>
					<p class="go_login" >没有账号 <router-link to="/user/reg">立即注册</router-link></p>
				</div>
			</div>
		</div>
	</div>
</template>

<script>
export default {
  name: 'Login',
  data(){
    return {
        login_type: 0,
        username:"",
        password:"",
        remember:false,
    }
  },
  methods:{
    loginHandler(){
        // 用户密码账号登录
        this.$axios.post(`${this.$settings.HOST}/user/login/`,{
            username:this.username,
            password:this.password,
        }).then(response=>{
            console.log(response);
            if(this.remember){
                // 记住登录状态
                sessionStorage.removeItem("user_token");
                sessionStorage.removeItem("user_id");
                sessionStorage.removeItem("user_name");
                localStorage.user_token = response.data.token;
                localStorage.user_id = response.data.id;
                localStorage.user_name = response.data.username;
            }else{
                // 不记住登录状态
                localStorage.removeItem("user_token");
                localStorage.removeItem("user_id");
                localStorage.removeItem("user_name");
                sessionStorage.user_token = response.data.token;
                sessionStorage.user_id = response.data.id;
                sessionStorage.user_name = response.data.username;
            }

            this.uploadAvatar()
        }).catch(error=>{
            this.$message.error("对不起，登录失败！请确认密码或账号是否正确！");
        });
    },
     // 获取用户昵称
     uploadAvatar() {
      if (localStorage.user_name || sessionStorage.user_name) {
        this.$axios
          .get(`${this.$settings.HOST}/user/person/${this.username}/`, {
            nickname: this.nickname,
          }).then(res => {
            console.log(res);
            localStorage.nickname = res.data.nickname
            sessionStorage.nickname = res.data.nickname
            localStorage.avatar = res.data.avatar
            sessionStorage.avatar = res.data.avatar
                let self = this;
            this.$alert("登录成功!","大头Chatbot",{
               callback(){
                    self.$router.push("/");
               }
            });
          })
          .catch(error => {
            console.log(error);
          });
      }
    },
  },

};
</script>

<style scoped>
.box{
	width: 100%;
  height: 100%;
	position: fixed;
  background-color:#b5c6c6;
}

.box .login {
	position: absolute;
	width: 500px;
	height: 300px;
	left: 0;
  margin: auto;
  right: 0;
  bottom: 0;
  top: -338px;
}
.login .login-title{
     width: 100%;
    text-align: center;
}
.login-title img{
    width: 190px;
    height: auto;
}
.login-title p{
    font-family: PingFangSC-Regular;
    font-size: 18px;
    color: #fff;
    letter-spacing: .29px;
    padding-top: 10px;
    padding-bottom: 50px;
}
.login_box{
    width: 400px;
    height: auto;
    background: #fff;
    box-shadow: 0 2px 4px 0 rgba(0,0,0,.5);
    border-radius: 4px;
    margin: 0 auto;
    padding-bottom: 40px;
    margin-top: 100px;
}
.login_box .title{
	font-size: 20px;
	color: #9b9b9b;
	letter-spacing: .32px;
	border-bottom: 1px solid #e6e6e6;
	 display: flex;
    	justify-content: space-around;
    	padding: 50px 42px 0 42px;
    	margin-bottom: 40px;
    	cursor: pointer;
}
.login_box .title span:nth-of-type(1){
	color: #4a4a4a;
    	border-bottom: 2px solid #88c0d0;
}
.login_box .title span:nth-of-type(2){
	color: #4a4a4a;
    	border-bottom: 2px solid #88c0d0;
}

.inp{
	width: 350px;
	margin: 0 auto;
}
.inp input{
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
.inp input.user{
    margin-bottom: 16px;
}
.inp .remember{
     display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    margin-top: 10px;
}
.inp .remember p:first-of-type{
    font-size: 12px;
    color: #4a4a4a;
    letter-spacing: .19px;
    margin-left: 22px;
    display: -ms-flexbox;
    display: flex;
    -ms-flex-align: center;
    align-items: center;
    /*position: relative;*/
}
.inp .remember p:nth-of-type(2){
    font-size: 14px;
    color: #9b9b9b;
    letter-spacing: .19px;
    cursor: pointer;
}

.inp .remember input{
    outline: 0;
    width: 30px;
    height: 45px;
    border-radius: 4px;
    border: 1px solid #d9d9d9;
    text-indent: 20px;
    font-size: 14px;
    background: #fff !important;
}

.inp .remember .no{
  width:20px;
}

.inp .remember p span{
  display: inline-block;
  font-size: 14px;
  width: 100px;
  color: #4a4a4a;
  padding-left: 5px;
  /*position: absolute;*/
/*left: 20px;*/

}
#geetest{
	margin-top: 20px;
}
.login_btn{
    width: 100%;
    height: 45px;
    background: #88c0d0;
    border-radius: 5px;
    font-size: 16px;
    color: #fff;
    letter-spacing: .26px;
    margin-top: 5px;
}
.inp .go_login{
    text-align: center;
    font-size: 14px;
    color: #9b9b9b;
    letter-spacing: .26px;
    padding-top: 20px;
}
.inp .go_login span{
    color: #84cc39;
    cursor: pointer;
}
</style>
