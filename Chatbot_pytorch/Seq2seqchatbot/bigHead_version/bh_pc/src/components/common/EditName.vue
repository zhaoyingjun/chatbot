<template>
    <div>
        <el-dialog title="编辑用户名" :visible="data.status" width="500px"  @close="resetForm">
  <el-form :model="data" :rules="rules" ref="formData">
    <el-form-item label="旧名称" label-width="65px" >
      <el-input v-model="data.oldName"  autocomplete="off" :disabled="true"></el-input>
    </el-form-item>
    <el-form-item label="新名称" label-width="65px" prop="name">
      <el-input v-model="data.form.name" placeholder="请输入名称" @keyup.native = "ent" ></el-input>
    </el-form-item>
  </el-form>
  <div slot="footer" class="dialog-footer" >
    <el-button @click="resetForm" >取 消</el-button>
    <el-button type="primary" @click="dialogFormVisible">确 定</el-button>
  </div>
</el-dialog>
    </div>
</template>
<script>    
export default{
    name:'EditNname',
    data(){
        return{
            data:{
                status:false,
                oldName:'',
                form:{
                    name:''
                }
            },
            rules:{
                name:[
                { required: true, message: '请输入名称', trigger: 'blur' },
                ]
            }
        }
    },
    props:{
        editData:{
            type:Object,
            required:true
        }
    },
    watch:{
        editData:function (n,o){
            this.data = {
                status:n.status,
                oldName:n.username,
                form:{
                    name:''
                }
            }
        }
    },
    methods:{
        ent(e){
            if(e.keyCode == 13){
                
                this.dialogFormVisible()
            }
            // console.log(e);
        },
        dialogFormVisible(){
        //     this.$refs.formData.validate((valid) => {
        //         console.log(valid);
        //   if (valid) {
        //     this.$emit('editData',this.data.form)
        //   } else {
            //     this.$message.warning('请输入名称')
            //     return false;
            //   }
            // });
            if(this.data.form.name == '') return this.$message.warning('内容不能为空')
            this.$emit('editData',this.data.form)
            this.resetForm()
        },
        resetForm() {
        this.data.status = false
        this.$refs.formData.resetFields();
      } 
    }
}
</script>
<style lang="less" scoped>
.el-dialog{
    border-radius: 15px;
}
.dialog-footer{
    .el-button--primary{
        background-color: #49565d;
        border:none;
        color: #fff;
    }
    .el-button--default{
        background-color: #49565d;
        color: #fff;
    }
}
</style>